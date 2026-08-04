#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
 EdgeAI Inertial Guide Glasses - Core PoC Skeleton
==============================================================================

本程式碼為開源專案 邱炳坤 計畫之一部分，採用 GPL-3.0 授權。

Copyright (C) 2026  二兒子

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

專案名稱：The Project - EdgeAI Inertial Guide Glasses
目標平台：無鏡頭、無網路 HUD 顯示眼鏡（如 Even G2 類型）
開發階段：v0.1 PoC（概念驗證骨架）

作者說明：
    本骨架專為銀髮族離線照護設計，所有運算均在邊緣端完成，
    不依賴 GPS、鏡頭或雲端連線。模組一提供慣性盲算導航；
    模組二提供情境感知服藥提醒。
==============================================================================
"""

import math
import time
import random
from collections import deque
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Tuple, Optional, Dict

# =============================================================================
# 全域常數與型態定義
# =============================================================================

class ActivityState(Enum):
    """使用者活動狀態列舉（用於服藥提醒模組的情境判斷）"""
    SLEEPING = "sleeping"           # 睡眠中：極低活動度
    WALKING = "walking"             # 行走中：規律步態、持續晃動
    STABLE = "stable"               # 平穩站立/坐姿：適合接收提醒
    UNKNOWN = "unknown"             # 未知狀態

class NavCommand(Enum):
    """導航視覺指令列舉（回傳給 HUD 顯示）"""
    GO_STRAIGHT = "⬆ 直走"
    TURN_LEFT = "⬅ 向左轉"
    TURN_RIGHT = "➡ 向右轉"
    TURN_BACK = "⬇ 往回走"
    ARRIVED = "🏠 已到家"
    CALIBRATING = "⟳ 校準中"


# =============================================================================
# 模組一：離線慣性盲算導航模組 (Dead Reckoning Algorithm)
# =============================================================================

class InertialDeadReckoning:
    """
    離線慣性盲算導航核心類別

    功能：
        1. 接收模擬 IMU 數據（Acc x/y/z, Gyro x/y/z）
        2. 以峰值偵測法（Peak Detection）計算步數
        3. 以加速度變異量估算步長（Weinberg 簡化模型）
        4. 以陀螺儀 Z 軸積分計算偏航角（Yaw）
        5. 累積相對位移，並與預存回家路線比對
        6. 若偏離超過閾值，產生轉向指令供 HUD 顯示

    注意：
        本 PoC 採用簡化物理模型。實際硬體部署時，建議加入：
        - 磁力計輔助航向校正（Magnetometer fusion）
        - 零速更新（ZUPT）抑制漂移
        - 卡爾曼濾波器（EKF）融合多軸數據
    """

    def __init__(
        self,
        home_route: List[Tuple[float, float]],
        step_threshold: float = 1.15,          # 步態偵測加速度閾值 (g)
        step_length_default: float = 0.65,     # 預設步長 (m)
        deviation_threshold: float = 8.0,      # 偏離路線閾值 (m)
        heading_drift_compensation: float = 0.02,  # 陀螺儀漂移補償 (rad/sample)
        window_size: int = 5                   # 平滑濾波窗格大小
    ):
        """
        初始化導航器

        Args:
            home_route: 預存回家路線座標陣列 [(x0,y0), (x1,y1), ...]，單位：公尺
            step_threshold: 判定為一步的加速度峰值閾值（以重力加速度 g 為單位）
            step_length_default: 預設步長，當無法估算時使用
            deviation_threshold: 偏離路線的最大容許距離（公尺）
            heading_drift_compensation: 每筆樣本的陀螺儀零偏補償量
            window_size: 移動平均濾波窗格大小
        """
        # --- 路線與閾值參數 ---
        self.home_route = home_route            # 預設回家路徑（相對座標系）
        self.step_threshold = step_threshold
        self.step_length_default = step_length_default
        self.deviation_threshold = deviation_threshold
        self.heading_drift_compensation = heading_drift_compensation

        # --- 狀態變數 ---
        self.position = [0.0, 0.0]              # 當前相對位移 [x, y]（公尺）
        self.heading = 0.0                      # 當前偏航角（弧度，0 = 正北/初始方向）
        self.total_steps = 0                    # 累計步數
        self.is_stepping = False                # 是否正處於單步週期中（防止重複計步）

        # --- 濾波與緩衝 ---
        self.window_size = window_size
        self.acc_buffer = deque(maxlen=window_size)   # 加速度平滑緩衝
        self.gyro_buffer = deque(maxlen=window_size)  # 陀螺儀平滑緩衝

        # --- 步長估算用變數 ---
        self.last_step_acc_variance = 0.0       # 上一步週期內的加速度變異量
        self.step_acc_history = []              # 單步週期內的加速度樣本

        # --- 時間相關 ---
        self.last_step_time = None
        self.sample_dt = 0.05                   # 預設取樣間隔 50ms (20Hz)

        print("[DeadReckoning] 導航模組初始化完成")
        print(f"[DeadReckoning] 預設回家路線共 {len(home_route)} 個路徑點")

    def _smooth(self, buffer: deque, new_val: float) -> float:
        """簡易移動平均濾波（Moving Average Filter）"""
        buffer.append(new_val)
        return sum(buffer) / len(buffer)

    def _calculate_step_length(self, acc_variance: float) -> float:
        """
        基於加速度變異量估算步長（Weinberg 簡化模型）

        原理：步長與垂直加速度的變異量呈正相關。
        公式：step_length = K * sqrt(acc_variance)
        其中 K 為經驗常數，此處簡化為 0.3。

        Args:
            acc_variance: 單步週期內三軸加速度向量的變異量

        Returns:
            估算步長（公尺）
        """
        k_weinberg = 0.3
        estimated = k_weinberg * math.sqrt(max(acc_variance, 0.01))
        # 限制在合理範圍內（銀髮族步長通常 0.4 ~ 0.8m）
        return max(0.4, min(estimated, 0.8))

    def _detect_nearest_route_point(self) -> Tuple[int, float]:
        """
        尋找當前位置在回家路線上的最近路徑點索引與距離

        Returns:
            (nearest_idx, min_distance)
        """
        min_dist = float('inf')
        nearest_idx = 0
        for i, (rx, ry) in enumerate(self.home_route):
            dist = math.hypot(self.position[0] - rx, self.position[1] - ry)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        return nearest_idx, min_dist

    def _compute_navigation_command(self) -> NavCommand:
        """
        根據當前位置與預設路線的偏離程度，計算導航指令

        邏輯：
            1. 找到最近路徑點
            2. 若距離 < 閾值：判斷前進方向是否與路線切線方向一致
            3. 若距離 >= 閾值：計算回到路線所需的轉向角
        """
        if not self.home_route:
            return NavCommand.CALIBRATING

        nearest_idx, min_dist = self._detect_nearest_route_point()

        # 若已抵達終點（最後一個路徑點且距離夠近）
        if nearest_idx == len(self.home_route) - 1 and min_dist < 3.0:
            return NavCommand.ARRIVED

        # 若偏離過大，計算回到路線的方向
        if min_dist > self.deviation_threshold:
            target_x, target_y = self.home_route[min(nearest_idx + 1, len(self.home_route) - 1)]
            dx = target_x - self.position[0]
            dy = target_y - self.position[1]
            desired_heading = math.atan2(dy, dx)

            # 計算當前航向與期望航向的差角（標準化到 -pi ~ pi）
            angle_diff = desired_heading - self.heading
            angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

            if abs(angle_diff) < math.radians(30):
                return NavCommand.GO_STRAIGHT
            elif angle_diff > 0:
                return NavCommand.TURN_LEFT
            else:
                return NavCommand.TURN_RIGHT

        # 偏離在容許範圍內，沿著路線前進
        next_idx = min(nearest_idx + 1, len(self.home_route) - 1)
        if next_idx == nearest_idx:
            return NavCommand.ARRIVED

        target_x, target_y = self.home_route[next_idx]
        dx = target_x - self.position[0]
        dy = target_y - self.position[1]
        desired_heading = math.atan2(dy, dx)

        angle_diff = desired_heading - self.heading
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

        if abs(angle_diff) < math.radians(20):
            return NavCommand.GO_STRAIGHT
        elif angle_diff > 0:
            return NavCommand.TURN_LEFT
        else:
            return NavCommand.TURN_RIGHT

    def process_imu_sample(
        self,
        acc: Tuple[float, float, float],   # (ax, ay, az) in g
        gyro: Tuple[float, float, float],  # (gx, gy, gz) in deg/s
        dt: Optional[float] = None
    ) -> Dict:
        """
        處理單筆 IMU 樣本，更新位置與航向，並回傳導航狀態

        Args:
            acc: 三軸加速度計讀值 (g)
            gyro: 三軸陀螺儀讀值 (deg/s)
            dt: 取樣間隔（秒），若未提供則使用預設值

        Returns:
            dict 包含：
                - 'position': [x, y] 當前相對座標
                - 'heading': 當前偏航角（度）
                - 'steps': 累計步數
                - 'step_length': 上一步步長
                - 'nav_command': NavCommand 導航指令
                - 'deviation': 與路線的偏離距離（公尺）
        """
        if dt is None:
            dt = self.sample_dt

        ax, ay, az = acc
        gx, gy, gz = gyro

        # --- 1. 濾波：對加速度與陀螺儀做移動平均 ---
        acc_magnitude = math.sqrt(ax**2 + ay**2 + az**2)
        smoothed_acc = self._smooth(self.acc_buffer, acc_magnitude)
        smoothed_gz = self._smooth(self.gyro_buffer, gz)

        # --- 2. 航向更新：積分陀螺儀 Z 軸（偏航角）---
        # 扣除零偏補償，轉換 deg/s → rad/sample
        gz_rad = math.radians(smoothed_gz - self.heading_drift_compensation)
        self.heading += gz_rad * dt
        # 標準化航向角至 0 ~ 2π
        self.heading = self.heading % (2 * math.pi)

        # --- 3. 步態偵測：峰值偵測法（Peak Detection）---
        step_detected = False
        step_length = 0.0

        # 收集單步週期內的加速度樣本（用於步長估算）
        self.step_acc_history.append(smoothed_acc)

        # 狀態機：偵測加速度峰值超過閾值，且前一狀態為非步態
        if smoothed_acc > self.step_threshold and not self.is_stepping:
            self.is_stepping = True
            step_detected = True
            self.total_steps += 1

            # 計算單步週期內的加速度變異量
            if len(self.step_acc_history) > 1:
                mean_acc = sum(self.step_acc_history) / len(self.step_acc_history)
                variance = sum((a - mean_acc) ** 2 for a in self.step_acc_history) / len(self.step_acc_history)
                self.last_step_acc_variance = variance
                step_length = self._calculate_step_length(variance)
            else:
                step_length = self.step_length_default

            # --- 4. 位移更新：根據步長與航向計算相對位移 ---
            dx = step_length * math.sin(self.heading)   # 東向分量
            dy = step_length * math.cos(self.heading)   # 北向分量
            self.position[0] += dx
            self.position[1] += dy

            # 重置單步歷史
            self.step_acc_history = []
            self.last_step_time = time.time()

        elif smoothed_acc < self.step_threshold * 0.85:
            # 加速度回落至基線以下，重置步態狀態
            self.is_stepping = False

        # --- 5. 導航指令計算 ---
        _, deviation = self._detect_nearest_route_point()
        nav_cmd = self._compute_navigation_command()

        return {
            'position': self.position.copy(),
            'heading': math.degrees(self.heading),
            'steps': self.total_steps,
            'step_length': step_length if step_detected else 0.0,
            'step_detected': step_detected,
            'nav_command': nav_cmd,
            'deviation': deviation,
            'raw_acc': acc_magnitude,
            'smoothed_acc': smoothed_acc
        }

    def reset(self):
        """重置所有狀態（用於重新開始導航）"""
        self.position = [0.0, 0.0]
        self.heading = 0.0
        self.total_steps = 0
        self.is_stepping = False
        self.acc_buffer.clear()
        self.gyro_buffer.clear()
        self.step_acc_history = []
        self.last_step_time = None
        print("[DeadReckoning] 導航狀態已重置")


# =============================================================================
# 模組二：邊緣 AI 情境感知吃藥提醒模組
# =============================================================================

class MedicationReminder:
    """
    邊緣 AI 情境感知服藥提醒核心類別

    功能：
        1. 設定每日服藥時程（如 08:00、12:30、18:00）
        2. 持續監測 IMU 數據，判斷使用者活動狀態
        3. 當服藥時間到達時，若使用者處於「睡眠」或「劇烈晃動（行走）」，
           自動延後觸發，避免打擾或造成危險
        4. 直到偵測為「平穩站立/坐姿」，才觸發 HUD 大字體通知
        5. 於終端機模擬輸出 2D 藥丸形狀與顏色視覺介面

    活動狀態判斷邏輯（邊緣端輕量規則，無需神經網路）：
        - SLEEPING: 加速度變異量 < 0.02g 且持續超過 30 秒
        - WALKING:  步態規律峰值且加速度變異量 > 0.3g
        - STABLE:   加速度變異量介於 0.02g ~ 0.15g 之間，無規律步態
    """

    def __init__(
        self,
        medication_times: List[str],       # 服藥時間列表 ["08:00", "12:30", ...]
        reminder_delay_minutes: int = 10,  # 每次延後觸發的間隔（分鐘）
        max_delays: int = 6,               # 最大延後次數（超過則強制提醒）
        sleep_threshold: float = 0.02,     # 睡眠判定加速度變異閾值 (g)
        walk_threshold: float = 0.30,      # 行走判定加速度變異閾值 (g)
        stable_window: int = 10            # 判定為穩定狀態所需的連續樣本數
    ):
        """
        初始化服藥提醒器

        Args:
            medication_times: 每日服藥時間字串列表（24小時制 HH:MM）
            reminder_delay_minutes: 條件不符時每次延後的分鐘數
            max_delays: 單次服藥提醒的最大延後次數
            sleep_threshold: 睡眠判定閾值（加速度變異量）
            walk_threshold: 行走判定閾值（加速度變異量）
            stable_window: 判定為穩定狀態所需的連續 IMU 樣本數
        """
        self.medication_times = sorted(medication_times)
        self.reminder_delay_minutes = reminder_delay_minutes
        self.max_delays = max_delays
        self.sleep_threshold = sleep_threshold
        self.walk_threshold = walk_threshold
        self.stable_window = stable_window

        # --- 狀態追蹤 ---
        self.pending_reminders: Dict[str, dict] = {}   # 待觸發的提醒 {time_str: info}
        self.activity_history = deque(maxlen=stable_window)  # 近期活動狀態緩衝
        self.current_activity = ActivityState.UNKNOWN
        self.last_variance = 0.0

        # --- 藥物視覺資料庫（藥丸名稱 → (形狀, 顏色 ANSI, 顯示字元)）---
        self.pill_database = {
            "血壓藥":   ("round", "\033[91m", "●"),      # 紅色圓形
            "降血糖藥": ("oval",  "\033[94m", "◉"),      # 藍色橢圓
            "維他命":   ("round", "\033[93m", "◎"),      # 黃色圓形
            "心臟藥":   ("capsule", "\033[96m", "▣"),    # 青色膠囊
        }

        print("[MedReminder] 服藥提醒模組初始化完成")
        print(f"[MedReminder] 每日服藥時程: {', '.join(self.medication_times)}")

    def _detect_activity(self, acc_variance: float, step_detected: bool) -> ActivityState:
        """
        輕量級活動狀態判斷（邊緣端規則引擎）

        Args:
            acc_variance: 近期加速度變異量
            step_detected: 是否偵測到步態（來自導航模組的步態訊號）

        Returns:
            ActivityState 列舉值
        """
        if acc_variance < self.sleep_threshold:
            return ActivityState.SLEEPING
        elif step_detected and acc_variance > self.walk_threshold:
            return ActivityState.WALKING
        elif acc_variance < self.walk_threshold * 0.5:
            return ActivityState.STABLE
        else:
            return ActivityState.UNKNOWN

    def _check_medication_schedule(self, current_time: datetime) -> List[str]:
        """
        檢查當前時間是否有待觸發的服藥提醒

        Returns:
            已到達觸發時間的藥物時間字串列表
        """
        triggered = []
        current_str = current_time.strftime("%H:%M")

        for med_time in self.medication_times:
            # 若尚未建立今日提醒記錄，則初始化
            if med_time not in self.pending_reminders:
                # 設定今日理論觸發時間
                hour, minute = map(int, med_time.split(':'))
                scheduled = current_time.replace(hour=hour, minute=minute, second=0, microsecond=0)
                if scheduled < current_time:
                    # 若已過時間，設定為明日（PoC 中簡化處理）
                    scheduled += timedelta(days=1)
                self.pending_reminders[med_time] = {
                    'scheduled': scheduled,
                    'delay_count': 0,
                    'triggered': False
                }

            info = self.pending_reminders[med_time]
            if not info['triggered'] and current_time >= info['scheduled']:
                triggered.append(med_time)

        return triggered

    def _render_hud_notification(self, med_time: str, pill_name: str):
        """
        模擬 HUD 大字體服藥通知與終端機 2D 藥丸視覺介面

        在實際硬體上，此函數應透過 UART/SPI 將指令傳送至 HUD 控制器。
        """
        pill_info = self.pill_database.get(pill_name, ("round", "\033[97m", "●"))
        shape, color_code, char = pill_info

        # 清除終端機並輸出 HUD 模擬畫面
        print("\n" + "=" * 50)
        print("         🔔 服藥提醒 HUD 模擬畫面")
        print("=" * 50)
        print(f"\n   現在時間: {med_time}")
        print(f"   請服用: {pill_name}")
        print("\n   ┌─────────────────────────────┐")
        print("   │                             │")
        print(f"   │        {color_code}{char}{char}{char}  {char}{char}{char}\033[0m        │")
        print(f"   │        {color_code}{char} 服藥時間 {char}\033[0m        │")
        print(f"   │        {color_code}{char}{char}{char}  {char}{char}{char}\033[0m        │")
        print("   │                             │")
        print("   └─────────────────────────────┘")
        print("\n   💡 請保持坐姿，確認藥名後服用")
        print("=" * 50 + "\n")

    def process_tick(
        self,
        current_time: datetime,
        acc_variance: float,
        step_detected: bool,
        pill_assignment: Optional[Dict[str, str]] = None
    ) -> List[Dict]:
        """
        主循環：處理每個時間刻度的狀態更新與提醒判斷

        Args:
            current_time: 當前系統時間
            acc_variance: 近期加速度變異量（來自 IMU）
            step_detected: 是否偵測到步態（來自導航模組）
            pill_assignment: 時間→藥名對照表，如 {"08:00": "血壓藥"}

        Returns:
            本時間刻度觸發的所有提醒事件列表
        """
        if pill_assignment is None:
            pill_assignment = {t: "血壓藥" for t in self.medication_times}

        triggered_events = []

        # --- 1. 更新活動狀態 ---
        activity = self._detect_activity(acc_variance, step_detected)
        self.activity_history.append(activity)
        self.current_activity = activity
        self.last_variance = acc_variance

        # 判斷是否達到「穩定狀態」的連續樣本要求
        is_stable_sustained = (
            len(self.activity_history) >= self.stable_window and
            all(a == ActivityState.STABLE for a in self.activity_history)
        )

        # --- 2. 檢查服藥時程 ---
        due_times = self._check_medication_schedule(current_time)

        for med_time in due_times:
            info = self.pending_reminders[med_time]
            pill_name = pill_assignment.get(med_time, "藥物")

            # --- 3. 情境感知延後邏輯 ---
            should_trigger = False
            delay_reason = ""

            if activity == ActivityState.SLEEPING:
                delay_reason = "使用者睡眠中，延後提醒"
            elif activity == ActivityState.WALKING:
                delay_reason = "使用者行走中，延後提醒（安全考量）"
            elif is_stable_sustained or info['delay_count'] >= self.max_delays:
                should_trigger = True
            else:
                delay_reason = f"等待穩定狀態確認 ({len(self.activity_history)}/{self.stable_window})"

            if should_trigger:
                # 觸發提醒
                info['triggered'] = True
                self._render_hud_notification(med_time, pill_name)
                triggered_events.append({
                    'time': med_time,
                    'pill': pill_name,
                    'status': 'triggered',
                    'delays': info['delay_count'],
                    'activity': activity.value
                })
            else:
                # 延後觸發
                info['scheduled'] += timedelta(minutes=self.reminder_delay_minutes)
                info['delay_count'] += 1
                triggered_events.append({
                    'time': med_time,
                    'pill': pill_name,
                    'status': 'delayed',
                    'reason': delay_reason,
                    'next_try': info['scheduled'].strftime("%H:%M"),
                    'delay_count': info['delay_count']
                })

        return triggered_events

    def reset_daily_schedule(self, current_time: datetime):
        """重置每日服藥排程（建議在每日 00:00 呼叫）"""
        self.pending_reminders.clear()
        self.activity_history.clear()
        print(f"[MedReminder] 已重置 {current_time.strftime('%Y-%m-%d')} 服藥排程")


# =============================================================================
# 模擬資料產生器（用於 PoC 測試，無需實際硬體）
# =============================================================================

class IMUSimulator:
    """
    IMU 數據模擬器

    產生不同活動情境下的合成 IMU 數據，供 PoC 測試使用。
    """

    @staticmethod
    def generate_walking_data(steps: int = 20, noise_level: float = 0.05):
        """
        模擬行走時的 IMU 數據

        Returns:
            List[((ax,ay,az), (gx,gy,gz))]
        """
        data = []
        for i in range(steps * 10):  # 每步 10 個樣本
            t = i / 10.0
            # 模擬週期性步態加速度（垂直方向正弦波 + 重力）
            az = 1.0 + 0.4 * math.sin(2 * math.pi * t) + random.gauss(0, noise_level)
            ax = random.gauss(0, noise_level)
            ay = random.gauss(0, noise_level)
            # 模擬輕微轉彎的陀螺儀讀值
            gz = 5.0 * math.sin(0.5 * t) + random.gauss(0, 2)
            gx = random.gauss(0, 1)
            gy = random.gauss(0, 1)
            data.append(((ax, ay, az), (gx, gy, gz)))
        return data

    @staticmethod
    def generate_sleeping_data(duration: int = 50, noise_level: float = 0.01):
        """模擬睡眠時的 IMU 數據（幾乎無晃動）"""
        data = []
        for _ in range(duration):
            ax = random.gauss(0, noise_level)
            ay = random.gauss(0, noise_level)
            az = 1.0 + random.gauss(0, noise_level)  # 靜止時僅有重力
            gx = random.gauss(0, 0.5)
            gy = random.gauss(0, 0.5)
            gz = random.gauss(0, 0.5)
            data.append(((ax, ay, az), (gx, gy, gz)))
        return data

    @staticmethod
    def generate_stable_data(duration: int = 30, noise_level: float = 0.03):
        """模擬平穩坐/站時的 IMU 數據（輕微身體晃動）"""
        data = []
        for _ in range(duration):
            ax = random.gauss(0, noise_level)
            ay = random.gauss(0, noise_level)
            az = 1.0 + random.gauss(0, noise_level)
            gx = random.gauss(0, 1)
            gy = random.gauss(0, 1)
            gz = random.gauss(0, 0.5)
            data.append(((ax, ay, az), (gx, gy, gz)))
        return data


# =============================================================================
# 主程式：PoC 整合演示
# =============================================================================

def main():
    """
    主演示函數：整合導航模組與服藥提醒模組的 PoC 流程

    模擬情境：
        1. 使用者從公園出發，沿預設路線走回家（模組一演示）
        2. 途中 08:00 服藥時間到達，但使用者正在行走，系統延後
        3. 使用者到家後坐下休息，系統偵測穩定狀態，觸發 HUD 服藥提醒
    """
    print("\n" + "=" * 60)
    print("  The Project - EdgeAI Inertial Guide Glasses")
    print("  v0.1 PoC 整合演示")
    print("=" * 60 + "\n")

    # -------------------------------------------------------------------------
    # 初始化：設定回家路線（相對座標，單位：公尺）
    # -------------------------------------------------------------------------
    # 模擬一條簡單的 L 型回家路線：先直走 30m，右轉，再直走 20m
    home_route = []
    for i in range(30):
        home_route.append((0.0, float(i)))      # 向北直走 30m
    for i in range(1, 21):
        home_route.append((float(i), 30.0))     # 向東轉彎走 20m

    navigator = InertialDeadReckoning(
        home_route=home_route,
        step_threshold=1.12,
        deviation_threshold=5.0
    )

    # -------------------------------------------------------------------------
    # 初始化：服藥提醒模組
    # -------------------------------------------------------------------------
    reminder = MedicationReminder(
        medication_times=["08:00"],
        reminder_delay_minutes=5,
        max_delays=3,
        stable_window=8
    )

    pill_schedule = {"08:00": "血壓藥"}

    # -------------------------------------------------------------------------
    # 模擬情境 A：行走中（導航 + 服藥時間到達但延後）
    # -------------------------------------------------------------------------
    print("\n【情境 A】使用者行走中，服藥時間 08:00 到達...\n")

    sim_time = datetime(2026, 8, 4, 7, 59, 0)
    walking_data = IMUSimulator.generate_walking_data(steps=25, noise_level=0.04)

    for i, (acc, gyro) in enumerate(walking_data):
        sim_time += timedelta(seconds=0.05)  # 20Hz 取樣

        # 導航模組處理
        nav_result = navigator.process_imu_sample(acc, gyro, dt=0.05)

        # 服藥提醒模組處理（每 5 個樣本檢查一次，避免過度輸出）
        if i % 5 == 0:
            # 計算近期加速度變異量（簡化：使用原始加速度模值與 1g 的差）
            ax, ay, az = acc
            acc_var = abs(math.sqrt(ax**2 + ay**2 + az**2) - 1.0)

            events = reminder.process_tick(
                current_time=sim_time,
                acc_variance=acc_var,
                step_detected=nav_result['step_detected'],
                pill_assignment=pill_schedule
            )

            for evt in events:
                if evt['status'] == 'delayed':
                    print(f"  [{sim_time.strftime('%H:%M:%S')}] ⏸ {evt['reason']} → 延至 {evt['next_try']} (第 {evt['delay_count']} 次延後)")

        # 每 20 步輸出一次導航狀態
        if nav_result['step_detected'] and nav_result['steps'] % 5 == 0:
            print(f"  [{sim_time.strftime('%H:%M:%S')}] 步數:{nav_result['steps']:3d} | "
                  f"位置:({nav_result['position'][0]:5.1f}, {nav_result['position'][1]:5.1f}) | "
                  f"航向:{nav_result['heading']:6.1f}° | 指令: {nav_result['nav_command'].value}")

    # -------------------------------------------------------------------------
    # 模擬情境 B：到家後坐下休息（穩定狀態，觸發提醒）
    # -------------------------------------------------------------------------
    print("\n【情境 B】使用者到家，坐下休息，系統偵測穩定狀態...\n")

    stable_data = IMUSimulator.generate_stable_data(duration=40, noise_level=0.02)

    for i, (acc, gyro) in enumerate(stable_data):
        sim_time += timedelta(seconds=0.05)

        nav_result = navigator.process_imu_sample(acc, gyro, dt=0.05)

        ax, ay, az = acc
        acc_var = abs(math.sqrt(ax**2 + ay**2 + az**2) - 1.0)

        events = reminder.process_tick(
            current_time=sim_time,
            acc_variance=acc_var,
            step_detected=nav_result['step_detected'],
            pill_assignment=pill_schedule
        )

        for evt in events:
            if evt['status'] == 'triggered':
                print(f"  [{sim_time.strftime('%H:%M:%S')}] ✅ 條件符合！活動狀態: {evt['activity']}，延後 {evt['delays']} 次後觸發提醒")
            elif evt['status'] == 'delayed':
                print(f"  [{sim_time.strftime('%H:%M:%S')}] ⏸ {evt['reason']} → 延至 {evt['next_try']}")

    # -------------------------------------------------------------------------
    # 模擬情境 C：睡眠中（再次延後示範）
    # -------------------------------------------------------------------------
    print("\n【情境 C】夜間睡眠中，服藥時間到達（演示延後邏輯）...\n")

    reminder2 = MedicationReminder(
        medication_times=["22:00"],
        reminder_delay_minutes=10,
        max_delays=2
    )

    sim_time = datetime(2026, 8, 4, 22, 0, 0)
    sleep_data = IMUSimulator.generate_sleeping_data(duration=60, noise_level=0.005)

    for i, (acc, gyro) in enumerate(sleep_data):
        sim_time += timedelta(seconds=0.05)

        ax, ay, az = acc
        acc_var = abs(math.sqrt(ax**2 + ay**2 + az**2) - 1.0)

        events = reminder2.process_tick(
            current_time=sim_time,
            acc_variance=acc_var,
            step_detected=False
        )

        for evt in events:
            if evt['status'] == 'delayed':
                print(f"  [{sim_time.strftime('%H:%M:%S')}] ⏸ {evt['reason']} → 延至 {evt['next_try']}")
            elif evt['status'] == 'triggered':
                print(f"  [{sim_time.strftime('%H:%M:%S')}] ⚠️ 已達最大延後次數，強制觸發提醒")

    # -------------------------------------------------------------------------
    # 總結輸出
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  PoC 演示結束")
    print("=" * 60)
    print(f"\n  導航統計：")
    print(f"    - 總步數: {navigator.total_steps}")
    print(f"    - 最終位置: ({navigator.position[0]:.2f}, {navigator.position[1]:.2f})")
    print(f"    - 最終航向: {math.degrees(navigator.heading):.1f}°")
    print(f"\n  服藥提醒統計：")
    print(f"    - 今日排程: {reminder.medication_times}")
    print(f"    - 最終活動狀態: {reminder.current_activity.value}")
    print("\n  感謝您對本開源專案的支持。")
    print("  願這份技術，能陪伴每一位長者平安回家。")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
