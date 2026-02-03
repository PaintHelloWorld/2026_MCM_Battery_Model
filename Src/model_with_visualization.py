import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd
import warnings
from scipy.stats import norm

warnings.filterwarnings('ignore')

# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 参数定义 ====================
C0 = 4500  # 初始电池容量 (mAh)
V_nominal = 3.8  # 标称电压 (V)
E_full_initial = C0 * V_nominal * 3600 / 1000  # 初始总能量 (J)


# ==================== 用户行为动态模型 ====================
class UserBehaviorModel:
    """基于马尔可夫链的用户行为动态模型"""

    def __init__(self, seed=42):
        np.random.seed(seed)
        self.current_state = 'light'
        self.last_update_hour = -1

        # 场景配置
        self.scenario_configs = {
            'light': {
                '4G': {'enabled': False, 'signal_strength': 'good'},
                '5G': {'enabled': False},
                'WiFi': {'enabled': False},
                'Bluetooth': {'enabled': False}
            },
            'typical': {
                '4G': {'enabled': True, 'signal_strength': 'good'},
                '5G': {'enabled': False},
                'WiFi': {'enabled': True, 'signal_strength': 'good'},
                'Bluetooth': {'enabled': False}
            },
            'heavy': {
                '4G': {'enabled': True, 'signal_strength': 'good'},
                '5G': {'enabled': True, 'signal_strength': 'good'},
                'WiFi': {'enabled': True, 'signal_strength': 'good'},
                'Bluetooth': {'enabled': True, 'signal_strength': 'good'}
            }
        }

        # 记录状态历史
        self.state_history = []

    @staticmethod
    def get_time_based_weights(hour):
        """根据时间段调整状态转移权重"""
        if 0 <= hour < 6:
            return {'light': 0.95, 'typical': 0.05, 'heavy': 0.00}
        elif 6 <= hour < 8:
            return {'light': 0.6, 'typical': 0.3, 'heavy': 0.1}
        elif 8 <= hour < 9:
            return {'light': 0.1, 'typical': 0.3, 'heavy': 0.6}
        elif 9 <= hour < 12:
            return {'light': 0.5, 'typical': 0.4, 'heavy': 0.1}
        elif 12 <= hour < 13:
            return {'light': 0.2, 'typical': 0.3, 'heavy': 0.5}
        elif 13 <= hour < 17:
            return {'light': 0.4, 'typical': 0.5, 'heavy': 0.1}
        elif 17 <= hour < 19:
            return {'light': 0.2, 'typical': 0.3, 'heavy': 0.5}
        elif 19 <= hour < 22:
            return {'light': 0.3, 'typical': 0.5, 'heavy': 0.2}
        else:
            return {'light': 0.8, 'typical': 0.2, 'heavy': 0.0}

    def update_state(self, t):
        """根据时间更新用户状态"""
        hour = int((t / 3600) % 24)

        if hour == self.last_update_hour:
            return self.current_state

        self.last_update_hour = hour
        time_weights = self.get_time_based_weights(hour)

        states = list(time_weights.keys())
        weights = list(time_weights.values())
        new_state = np.random.choice(states, p=weights)

        self.current_state = new_state
        self.state_history.append((t, hour, new_state))

        return new_state

    def get_network_config(self):
        """获取当前状态的网络配置"""
        return self.scenario_configs[self.current_state].copy()

    def get_state_history(self):
        """获取状态历史"""
        return pd.DataFrame(self.state_history, columns=['time', 'hour', 'state'])

# ==================== 老化函数 ====================
def f_aging(cycle_count, alpha=0.0015):
    """指数衰减模型"""
    base_decay = np.exp(-alpha * cycle_count)
    aging_factor = 0.8 + 0.2 * (base_decay ** 0.3)
    return aging_factor

# ==================== 温度影响函数 ====================
def f_temp(T_b):
    """温度对电池容量的影响"""
    if T_b < -20:
        return 0.3
    elif T_b < 0:
        return 0.4 + 0.03 * (T_b + 20)
    elif T_b < 10:
        return 0.85 + 0.015 * T_b
    elif T_b < 40:
        return 1.0
    elif T_b < 50:
        return 1.0 - 0.015 * (T_b - 40)
    else:
        return 0.75 - 0.01 * (T_b - 50)

# ==================== 温度变化函数 ====================
def T_battery(t, ambient=25, usage_factor=0.01):
    """电池温度变化"""
    temperature = ambient + usage_factor * t / 3600 + 0.1 * np.sin(2 * np.pi * t / 1800)
    return max(min(temperature, 50), 0)

# ==================== 温度对功耗影响 ====================
def temperature_effect_on_power(temp, component='general'):
    """温度对电子元件功耗的影响"""
    if component == 'cpu':
        if temp < 0:
            return 1.3
        elif temp < 20:
            return 1.0 + 0.015 * (20 - temp)
        elif temp < 40:
            return 1.0
        elif temp < 60:
            return 1.0 + 0.02 * (temp - 40)
        else:
            return 1.5
    elif component == 'screen':
        if temp < -10:
            return 1.2
        elif temp > 50:
            return 1.1
        else:
            return 1.0
    elif component == 'network':
        if temp < 5:
            return 1.25
        elif temp > 45:
            return 1.15
        else:
            return 1.0
    else:
        delta_T = temp - 25
        return 1.0 + 0.005 * delta_T + 0.0001 * delta_T ** 2

# ==================== 综合网络功耗函数 ====================
def network_power_comprehensive(t, scenario, battery_temp, network_config):
    """计算网络总功耗"""
    base_powers = {
        '4G': {'idle': 0.05, 'active': 0.4},
        '5G': {'idle': 0.08, 'active': 0.6},
        'WiFi': {'idle': 0.01, 'active': 0.1},
        'Bluetooth': {'idle': 0.001, 'active': 0.02}
    }

    def get_signal_factor(strength):
        factors = {'poor': 2.0, 'fair': 1.5, 'good': 1.2, 'excellent': 1.0}
        return factors.get(strength, 1.2)

    t_mod = t % 600
    is_active = t_mod < 60

    total_power = 0
    for net_type in ['4G', '5G', 'WiFi', 'Bluetooth']:
        if network_config.get(net_type, {}).get('enabled', False):
            config = network_config[net_type]
            signal = config.get('signal_strength', 'good')

            base = base_powers[net_type]['active'] if is_active else base_powers[net_type]['idle']

            if scenario == 'heavy' and is_active:
                base *= 1.3
            elif scenario == 'light':
                base *= 0.7

            signal_factor = get_signal_factor(signal)
            temp_factor = 1.0 + 0.005 * (battery_temp - 25)
            power = base * signal_factor * temp_factor
            total_power += max(power, 0.001)

    return total_power

# ==================== 总功耗函数 ====================
def P_total(t, scenario='typical', battery_temp=25, params=None, user_model=None):
    """总功耗函数"""

    if user_model is not None:
        current_scenario = user_model.update_state(t)
        network_config = user_model.get_network_config()

        if params is None:
            params = {}
        params['scenario'] = current_scenario
        params['network_config'] = network_config
        scenario = current_scenario

    default_params = {
        'base_power': 0.6,
        'n_background_apps': 3,
        'gps_enabled': False,
        'screen_timeout': 300,
        'network_config': {
            '4G': {'enabled': True, 'signal_strength': 'good'},
            '5G': {'enabled': False},
            'WiFi': {'enabled': False},
            'Bluetooth': {'enabled': False}
        }
    }

    if params is not None:
        default_params.update(params)

    base_power = default_params['base_power']
    n_background_apps = default_params['n_background_apps']
    gps_enabled = default_params['gps_enabled']
    screen_timeout = default_params['screen_timeout']
    network_config = default_params['network_config']

    # 1. 屏幕功耗
    def screen_power(t):
        screen_cycle_period = 1800
        t_mod = t % screen_cycle_period
        screen_on = t_mod < (screen_cycle_period / 3)

        if screen_on and (t_mod % screen_timeout) > (screen_timeout - 60):
            screen_on = False

        if scenario == 'light':
            base = 0.3 if screen_on else 0.03
        elif scenario == 'heavy':
            base = 1.2 if screen_on else 0.08
        else:
            base = 0.6 if screen_on else 0.05

        temp_factor = temperature_effect_on_power(battery_temp, 'screen')
        return base * temp_factor

    # 2. CPU功耗
    def cpu_power(t):
        base_freq = 1.0 if scenario == 'light' else 2.0 if scenario == 'heavy' else 1.5
        periodic_load = 0.3 * np.sin(2 * np.pi * t / 1200)

        random_peaks = 0
        if np.random.rand() < 0.001:
            random_peaks = 0.5 * np.exp(-(t % 100) / 20)

        background_load = 0.05 * n_background_apps
        total_load = base_freq + periodic_load + random_peaks + background_load
        cpu_base_power = 0.1 * total_load ** 2

        if scenario == 'light':
            cpu_base_power *= 0.7
        elif scenario == 'heavy':
            cpu_base_power *= 1.8

        temp_factor = temperature_effect_on_power(battery_temp, 'cpu')
        return max(cpu_base_power * temp_factor, 0.05)

    # 3. 网络功耗
    network_power_value = network_power_comprehensive(t, scenario, battery_temp, network_config)

    # 4. 后台任务功耗
    def background_power():
        power_per_app = 0.02
        periodic_activity = 0.01 * np.sin(2 * np.pi * t / 1800)
        total_background_power = n_background_apps * power_per_app + abs(periodic_activity)

        if scenario == 'heavy':
            total_background_power *= 1.5
        elif scenario == 'light':
            total_background_power *= 0.7

        return max(total_background_power, 0.01)

    # 5. GPS功耗
    def gps_power(t):
        if not gps_enabled:
            return 0.0

        gps_cycle_period = 7200
        t_mod = t % gps_cycle_period
        gps_active = t_mod < 1800

        if gps_active:
            if scenario == 'heavy':
                base = 0.15
            elif scenario == 'light':
                base = 0.08
            else:
                base = 0.12
        else:
            base = 0.01

        temp_factor = temperature_effect_on_power(battery_temp, 'gps')
        return base * temp_factor

    # 总功耗
    total = (base_power * temperature_effect_on_power(battery_temp, 'general') +
             screen_power(t) +
             cpu_power(t) +
             network_power_value +
             background_power() +
             gps_power(t))

    noise = 0.02 * np.random.randn()
    return max(total + noise, 0.1)

# ==================== SOC微分方程 ====================
def soc_ode(t, SOC, params, user_model=None, ambient_temp=25):
    """SOC微分方程（考虑内阻影响）"""
    SOC_current = max(min(SOC[0], 100), 0)

    if SOC_current <= 0.01:
        return [0]

    # 计算当前电池温度
    T_b = T_battery(t, ambient=ambient_temp)

    # 计算老化影响
    aging_factor = f_aging(params['cycle_count'], params['aging_alpha'])

    # 计算温度对电池容量的影响
    temp_capacity_factor = f_temp(T_b)

    # 计算当前有效总能量
    E_full = C0 * V_nominal * 3600 / 1000 * aging_factor * temp_capacity_factor

    if E_full <= 1e-10:
        E_full = 1e-10

    # 计算总功耗
    P_tot = P_total(t, battery_temp=T_b, params=params, user_model=user_model)

    # 内阻影响
    def add_internal_resistance_effect(power, SOC, T_b, R_internal=0.029):
        V_actual = V_nominal * (0.9 + 0.1 * SOC / 100)
        I = power / V_actual
        R_temp = R_internal * (1 + 0.01 * (T_b - 25))
        P_internal = I ** 2 * R_temp
        total_power = power + P_internal
        return total_power, I, P_internal

    P_tot_with_R, current, P_internal = add_internal_resistance_effect(P_tot, SOC_current, T_b, 0.029)

    # SOC微分方程
    dSOC_dt = -100 * P_tot_with_R / E_full

    if dSOC_dt < -50:
        dSOC_dt = -50

    return [dSOC_dt]

# ==================== 主模拟函数 ====================
def simulate_battery_drain(params, initial_SOC=100, t_end=24 * 3600,
                           dynamic_mode=False, user_model=None, ambient_temperature=25):
    """模拟电池放电过程"""

    def event_soc_low(t, SOC, *args):
        return SOC[0] - 0.1

    event_soc_low.terminal = True
    event_soc_low.direction = -1

    t_eval = np.linspace(0, t_end, min(5000, int(t_end / 100)))

    def ode_func_static(t, SOC, params, ambient_temp=ambient_temperature):
        return soc_ode(t, SOC, params, ambient_temp=ambient_temp)

    def ode_func_dynamic(t, SOC, params, user_model, ambient_temp=ambient_temperature):
        return soc_ode(t, SOC, params, user_model=user_model, ambient_temp=ambient_temp)

    if dynamic_mode and user_model is not None:
        sol = solve_ivp(
            ode_func_dynamic,
            [0, t_end],
            [initial_SOC],
            args=(params, user_model, ambient_temperature),
            t_eval=t_eval,
            events=[event_soc_low],
            rtol=1e-4,
            atol=1e-6,
            max_step=30,
            method='RK45'
        )
    else:
        sol = solve_ivp(
            ode_func_static,
            [0, t_end],
            [initial_SOC],
            args=(params, ambient_temperature),
            t_eval=t_eval,
            events=[event_soc_low],
            rtol=1e-4,
            atol=1e-6,
            max_step=30,
            method='RK45'
        )

    return sol

# ==================== 可视化类 ====================
class BatteryVisualizer:
    """Battery Model Visualization Class"""

    def __init__(self):
        self.results = {}

    def create_individual_charts(self, sol_dynamic, sol_static, user_model, time_to_empty):
        """Create individual charts instead of one large figure"""
        charts = []

        # 1. Main discharge curve
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        self.plot_main_discharge_curve(ax1, sol_dynamic, time_to_empty)
        plt.tight_layout()
        charts.append(('Main_Discharge_Curve', fig1))

        # 2. User behavior states
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        self.plot_user_behavior_states(ax2, user_model)
        plt.tight_layout()
        charts.append(('User_Behavior_States', fig2))

        # 3. Power composition analysis
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        self.plot_power_composition(ax3, sol_dynamic, user_model)
        plt.tight_layout()
        charts.append(('Power_Composition_Analysis', fig3))

        # 4. Temperature effects
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        self.plot_temperature_effects(ax4)
        plt.tight_layout()
        charts.append(('Temperature_Effects', fig4))

        # 5. Aging effects
        fig5, ax5 = plt.subplots(figsize=(10, 6))
        self.plot_aging_effects(ax5)
        plt.tight_layout()
        charts.append(('Aging_Effects', fig5))

        # 6. Scenario comparison
        fig6, ax6 = plt.subplots(figsize=(10, 6))
        self.plot_scenario_comparison(ax6, sol_static)
        plt.tight_layout()
        charts.append(('Scenario_Comparison', fig6))

        # 7. Network configuration impact
        fig7, ax7 = plt.subplots(figsize=(10, 6))
        self.plot_network_impact(ax7)
        plt.tight_layout()
        charts.append(('Network_Configuration_Impact', fig7))

        # 8. Daily usage pattern
        fig8, ax8 = plt.subplots(figsize=(10, 6))
        self.plot_daily_usage_pattern(ax8, user_model)
        plt.tight_layout()
        charts.append(('Daily_Usage_Pattern', fig8))

        # 10. New: 3D plot of power vs SOC and temperature
        fig10 = plt.figure(figsize=(12, 8))
        self.plot_power_soc_temp_3d(fig10)
        plt.tight_layout()
        charts.append(('Power_SOC_Temperature_3D', fig10))

        # 11. New: Typical user usage pattern normal distribution
        fig11, ax11 = plt.subplots(figsize=(10, 6))
        self.plot_typical_user_distribution(ax11)
        plt.tight_layout()
        charts.append(('User_Usage_Pattern_Distribution', fig11))

        # 12. New: Energy saving strategies comparison
        fig12, ax12 = plt.subplots(figsize=(10, 6))
        self.plot_energy_saving_strategies(ax12)
        plt.tight_layout()
        charts.append(('Energy_Saving_Strategies_Comparison', fig12))

        # Save all charts
        for name, fig in charts:
            filename = f'battery_{name}.png'
            fig.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Chart saved: {filename}")

        return charts

    def plot_main_discharge_curve(self, ax, sol, time_to_empty):
        """Plot main discharge curve"""
        hours = sol.t / 3600
        soc = sol.y[0] if sol.y.ndim > 1 else sol.y

        ax.plot(hours, soc, 'b-', linewidth=3, label='SOC')
        ax.axhline(y=20, color='orange', linestyle='--', alpha=0.7, label='20% Warning')
        ax.axhline(y=5, color='red', linestyle='--', alpha=0.7, label='5% Shutdown')

        if time_to_empty:
            ax.axvline(x=time_to_empty, color='green', linestyle=':',
                       label=f'Estimated Depletion: {time_to_empty:.1f}h')

        ax.set_xlabel('Time (hours)', fontsize=12)
        ax.set_ylabel('Battery Level (%)', fontsize=12)
        ax.set_title('Battery Discharge Curve', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        ax.set_ylim(0, 105)

        # Add endurance time annotation
        ax.text(0.02, 0.95, f'Endurance Time: {time_to_empty:.2f} hours',
                transform=ax.transAxes, fontsize=11,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def plot_user_behavior_states(self, ax, user_model):
        """Plot user behavior states"""
        if hasattr(user_model, 'state_history'):
            df_states = user_model.get_state_history()

            # Create state mapping
            state_map = {'light': 0, 'typical': 1, 'heavy': 2}
            state_values = [state_map[state] for state in df_states['state']]

            ax.step(df_states['time'] / 3600, state_values, where='post',
                    linewidth=2, color='darkblue')

            # Set y-axis labels
            ax.set_yticks([0, 1, 2])
            ax.set_yticklabels(['Light Usage', 'Typical Usage', 'Heavy Usage'])

            # Add time segments
            for hour in range(0, 25, 6):
                ax.axvline(x=hour, color='gray', linestyle=':', alpha=0.5)

            ax.set_xlabel('Time (hours)', fontsize=11)
            ax.set_title('User Behavior State Changes', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 24)
            ax.set_ylim(-0.5, 2.5)

            # Add time period annotations
            time_periods = ['Late Night', 'Morning', 'Morning Work', 'Lunch Break',
                          'Afternoon', 'Evening', 'Before Bed']
            for i, period in enumerate(time_periods):
                ax.text(i * 4 + 1, 2.2, period, ha='center', fontsize=9, alpha=0.8)

    def plot_power_composition(self, ax, sol, user_model):
        """Plot power composition analysis"""
        # Sample time points
        sample_times = np.linspace(0, min(24 * 3600, sol.t[-1]), 8)

        # Calculate power for each component
        components = ['Screen', 'CPU', 'Network', 'Background', 'Base', 'GPS']
        power_data = []

        for t in sample_times:
            hour = t / 3600

            # Get current scenario
            if user_model:
                scenario = user_model.update_state(t)
                network_config = user_model.get_network_config()
            else:
                scenario = 'typical'
                network_config = {
                    '4G': {'enabled': True, 'signal_strength': 'good'},
                    '5G': {'enabled': False},
                    'WiFi': {'enabled': False},
                    'Bluetooth': {'enabled': False}
                }

            # Simulate component power calculation
            T_b = T_battery(t)

            # Screen power
            screen_on = (t % 1800) < 600
            if scenario == 'light':
                screen_pwr = 0.3 if screen_on else 0.03
            elif scenario == 'heavy':
                screen_pwr = 1.2 if screen_on else 0.08
            else:
                screen_pwr = 0.6 if screen_on else 0.05

            # CPU power
            base_freq = 1.0 if scenario == 'light' else 2.0 if scenario == 'heavy' else 1.5
            periodic_load = 0.3 * np.sin(2 * np.pi * t / 1200)
            cpu_pwr = 0.1 * (base_freq + periodic_load) ** 2

            # Network power
            net_pwr = network_power_comprehensive(t, scenario, T_b, network_config)

            # Background power
            back_pwr = 0.06

            # Base power
            base_pwr = 0.6

            # GPS power
            gps_on = (t % 7200) < 1800
            gps_pwr = 0.12 if gps_on else 0.01

            power_data.append([screen_pwr, cpu_pwr, net_pwr, back_pwr, base_pwr, gps_pwr])

        power_data = np.array(power_data).T
        x_pos = np.arange(len(sample_times))
        width = 0.12

        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']

        bottom = np.zeros(len(sample_times))
        for i in range(len(components)):
            ax.bar(x_pos + i * width, power_data[i], width, label=components[i],
                   color=colors[i], bottom=bottom, alpha=0.8)
            bottom += power_data[i]

        ax.set_xlabel('Time Points (hours)', fontsize=11)
        ax.set_ylabel('Power (W)', fontsize=11)
        ax.set_title('Power Composition Analysis', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos + width * 2.5)
        ax.set_xticklabels([f'{t / 3600:.1f}' for t in sample_times])
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    def plot_temperature_effects(self, ax):
        """Plot temperature effects"""
        temps = np.linspace(-20, 60, 100)
        capacity_factors = [f_temp(t) for t in temps]

        ax.plot(temps, capacity_factors, 'r-', linewidth=2)
        ax.axvspan(15, 30, alpha=0.2, color='green', label='Optimal Range')
        ax.axvline(x=25, color='blue', linestyle='--', alpha=0.7, label='Reference Temperature')

        # Mark special temperature points
        special_temps = [-20, 0, 10, 25, 40, 50, 60]
        for temp in special_temps:
            factor = f_temp(temp)
            ax.plot(temp, factor, 'bo')
            ax.text(temp, factor + 0.02, f'{factor:.2f}', ha='center', fontsize=9)

        ax.set_xlabel('Temperature (°C)', fontsize=12)
        ax.set_ylabel('Relative Capacity', fontsize=12)
        ax.set_title('Temperature Effects on Battery Capacity', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(0.3, 1.2)

        # Add performance descriptions
        ax.text(0.02, 0.95, 'Extreme Cold: Capacity Drop', transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        ax.text(0.02, 0.85, '20-40°C: Optimal Performance', transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        ax.text(0.02, 0.75, 'High Temp: Capacity Degradation', transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

    def plot_aging_effects(self, ax):
        """Plot aging effects"""
        cycles = np.linspace(0, 2000, 100)
        aging_factors = [f_aging(c, 0.0015) for c in cycles]

        ax.plot(cycles, aging_factors, 'purple', linewidth=2)

        # Mark important cycle counts
        important_cycles = [0, 500, 1000, 1500, 2000]
        for cycle in important_cycles:
            factor = f_aging(cycle, 0.0015)
            ax.plot(cycle, factor, 'ro')
            ax.text(cycle, factor - 0.02, f'{cycle}', ha='center', fontsize=9)
            ax.text(cycle, factor + 0.02, f'{factor:.2f}', ha='center', fontsize=9)

        ax.set_xlabel('Cycle Count', fontsize=12)
        ax.set_ylabel('Health Factor', fontsize=12)
        ax.set_title('Battery Aging Curve', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.7, 1.05)

        # Add health level descriptions
        health_levels = [(0.9, 'Good'), (0.8, 'Fair'), (0.7, 'Poor')]
        for factor, label in health_levels:
            ax.axhline(y=factor, color='gray', linestyle=':', alpha=0.5)
            ax.text(cycles[-1] * 0.8, factor, label, ha='right', va='bottom', fontsize=9)

    def plot_scenario_comparison(self, ax, static_results):
        """Plot different scenario comparison"""
        scenarios = ['light', 'typical', 'heavy']
        colors = ['green', 'blue', 'red']

        for i, scenario in enumerate(scenarios):
            if scenario in static_results and static_results[scenario]['t'] is not None:
                hours = static_results[scenario]['t'] / 3600
                soc = static_results[scenario]['SOC']

                # Find time when battery drops to 5%
                below_5 = np.where(soc <= 5)[0]
                end_time = hours[-1] if len(below_5) == 0 else hours[below_5[0]]

                ax.plot(hours, soc, color=colors[i], linewidth=2,
                        label=f'{scenario} ({end_time:.1f}h)')

        ax.set_xlabel('Time (hours)', fontsize=12)
        ax.set_ylabel('Battery Level (%)', fontsize=12)
        ax.set_title('Usage Scenario Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        ax.set_ylim(0, 105)

        # Calculate endurance difference
        if 'light' in static_results and 'heavy' in static_results:
            light_time = static_results['light']['t'][-1] / 3600 if len(static_results['light']['t']) > 0 else 0
            heavy_time = static_results['heavy']['t'][-1] / 3600 if len(static_results['heavy']['t']) > 0 else 0
            if heavy_time > 0:
                diff_percent = (light_time - heavy_time) / heavy_time * 100
                ax.text(0.02, 0.95, f'Endurance Difference: {diff_percent:.0f}%',
                        transform=ax.transAxes, fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    def plot_network_impact(self, ax):
        """Plot network configuration impact"""
        network_configs = [
            {'name': '4G Only', 'config': {'4G': True, '5G': False, 'WiFi': False, 'BT': False}},
            {'name': '4G+WiFi', 'config': {'4G': True, '5G': False, 'WiFi': True, 'BT': False}},
            {'name': '5G+WiFi', 'config': {'4G': False, '5G': True, 'WiFi': True, 'BT': False}},
            {'name': 'All On', 'config': {'4G': True, '5G': True, 'WiFi': True, 'BT': True}},
            {'name': 'Airplane Mode', 'config': {'4G': False, '5G': False, 'WiFi': False, 'BT': False}}
        ]

        # Simulate endurance time estimation
        base_time = 12
        time_estimates = []

        for config in network_configs:
            power_multiplier = 1.0
            if config['name'] == '4G Only':
                power_multiplier = 1.0
            elif config['name'] == '4G+WiFi':
                power_multiplier = 1.15
            elif config['name'] == '5G+WiFi':
                power_multiplier = 1.3
            elif config['name'] == 'All On':
                power_multiplier = 1.5
            elif config['name'] == 'Airplane Mode':
                power_multiplier = 0.7

            time_estimate = base_time / power_multiplier
            time_estimates.append(time_estimate)

        bars = ax.bar(range(len(network_configs)), time_estimates,
                      color=['#4CAF50', '#2196F3', '#FF9800', '#F44336', '#9C27B0'])

        ax.set_xlabel('Network Configuration', fontsize=11)
        ax.set_ylabel('Estimated Endurance (hours)', fontsize=11)
        ax.set_title('Network Configuration Impact on Endurance', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(network_configs)))
        ax.set_xticklabels([cfg['name'] for cfg in network_configs], rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for i, (bar, time) in enumerate(zip(bars, time_estimates)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.2,
                    f'{time:.1f}h', ha='center', va='bottom', fontsize=9)

        # Add energy saving tip
        max_saving = (time_estimates[-1] - time_estimates[3]) / time_estimates[3] * 100
        ax.text(0.5, 0.9, f'Airplane Mode Saves\n{max_saving:.0f}%',
                transform=ax.transAxes, ha='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    def plot_daily_usage_pattern(self, ax, user_model):
        """Plot daily usage pattern"""
        hours = np.arange(0, 24, 1)

        # Define typical power consumption pattern
        base_pattern = np.array([
            0.2, 0.2, 0.2, 0.2, 0.2, 0.3,
            0.8, 1.5, 2.0, 2.0, 1.8, 1.5,
            1.2, 1.5, 1.8, 2.0, 2.0, 2.2,
            2.5, 2.0, 1.5, 1.0, 0.5, 0.3
        ])

        # Add random variation
        pattern = base_pattern * (1 + 0.1 * np.random.randn(24))
        pattern = np.maximum(pattern, 0.1)

        # Plot power consumption pattern
        ax.bar(hours, pattern, width=0.8, color='steelblue', alpha=0.7, edgecolor='darkblue')

        # Add usage scenario annotations
        usage_scenarios = [
            (0, 6, 'Sleep', 'lightblue'),
            (6, 9, 'Morning Activity', 'lightgreen'),
            (9, 12, 'Morning Work', 'gold'),
            (12, 14, 'Lunch Break', 'orange'),
            (14, 18, 'Afternoon Work', 'gold'),
            (18, 22, 'Evening Entertainment', 'lightcoral'),
            (22, 24, 'Before Bed', 'lavender')
        ]

        for start, end, label, color in usage_scenarios:
            ax.axvspan(start, end, alpha=0.2, color=color)
            ax.text((start + end) / 2, max(pattern) * 1.05, label,
                    ha='center', fontsize=8, rotation=45)

        ax.set_xlabel('Time of Day (hours)', fontsize=12)
        ax.set_ylabel('Typical Power (W)', fontsize=12)
        ax.set_title('Daily Typical Usage Pattern', fontsize=14, fontweight='bold', y = 1.15)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xlim(-0.5, 23.5)
        ax.set_xticks(range(0, 24, 3))

        # Add statistics
        avg_power = np.mean(pattern)
        peak_power = np.max(pattern)
        total_energy = np.sum(pattern)

        ax.text(0.02, 0.95, f'Average Power: {avg_power:.2f}W',
                transform=ax.transAxes, fontsize=9)
        ax.text(0.02, 0.85, f'Peak Power: {peak_power:.2f}W',
                transform=ax.transAxes, fontsize=9)
        ax.text(0.02, 0.75, f'Daily Total Energy: {total_energy:.1f}Wh',
                transform=ax.transAxes, fontsize=9)

    def plot_power_soc_temp_3d(self, fig):
        """Plot 3D relationship between power, SOC, and temperature"""
        ax = fig.add_subplot(111, projection='3d')

        # Generate data
        soc_range = np.linspace(10, 100, 20)
        temp_range = np.linspace(-10, 50, 20)
        soc_grid, temp_grid = np.meshgrid(soc_range, temp_range)

        # Calculate power
        base_power = 0.6
        soc_factor = 1.0 + 0.5 * (1 - soc_grid / 100)
        temp_factor = 1.0 + 0.005 * (temp_grid - 25) + 0.0001 * (temp_grid - 25) ** 2
        power_grid = base_power * soc_factor * temp_factor

        # Plot 3D surface
        surf = ax.plot_surface(soc_grid, temp_grid, power_grid, cmap='viridis',
                               alpha=0.8, linewidth=0, antialiased=True)

        # Add color bar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Power (W)')

        ax.set_xlabel('SOC (%)', fontsize=12, labelpad=10)
        ax.set_ylabel('Temperature (°C)', fontsize=12, labelpad=10)
        ax.set_zlabel('Power (W)', fontsize=12, labelpad=10)
        ax.set_title('Power vs SOC and Temperature (3D Surface)', fontsize=14, fontweight='bold', pad=20)

        # Adjust viewing angle
        ax.view_init(elev=25, azim=45)

        # Add description
        ax.text2D(0.02, 0.98, 'Observation: Power increases at low SOC and extreme temperatures',
                  transform=ax.transAxes, fontsize=10,
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def plot_typical_user_distribution(self, ax):
        """Plot normal distribution of typical user usage patterns"""
        # Define different user types
        user_types = ['Light User', 'Typical User', 'Heavy User', 'Gamer', 'Business User']

        # Define average daily screen time (hours) and standard deviation for each user type
        user_stats = {
            'Light User': {'mean': 3, 'std': 0.5},
            'Typical User': {'mean': 5, 'std': 1.0},
            'Heavy User': {'mean': 8, 'std': 1.5},
            'Gamer': {'mean': 6, 'std': 2.0},
            'Business User': {'mean': 7, 'std': 1.2}
        }

        # Create subplot
        colors = plt.cm.Set3(np.linspace(0, 1, len(user_types)))

        x = np.linspace(0, 14, 1000)

        # Plot normal distribution curve for each user type
        for i, user_type in enumerate(user_types):
            stats = user_stats[user_type]
            y = norm.pdf(x, stats['mean'], stats['std'])
            ax.plot(x, y, color=colors[i], linewidth=2, label=user_type)

            # Fill area under curve
            ax.fill_between(x, 0, y, color=colors[i], alpha=0.3)

            # Mark mean value
            ax.axvline(x=stats['mean'], color=colors[i], linestyle='--', alpha=0.5)
            ax.text(stats['mean'], norm.pdf(stats['mean'], stats['mean'], stats['std']) * 1.1,
                    f'{stats["mean"]}h', ha='center', fontsize=9)

        ax.set_xlabel('Daily Screen Time (hours)', fontsize=12)
        ax.set_ylabel('Probability Density', fontsize=12)
        ax.set_title('Typical User Usage Pattern Normal Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

        # Add statistics
        total_text = "User Type Distribution Statistics:\n"
        for user_type in user_types[:3]:
            stats = user_stats[user_type]
            total_text += f"{user_type}: {stats['mean']}±{stats['std']:.1f}h\n"

        ax.text(0.02, 0.98, total_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    def plot_energy_saving_strategies(self, ax):
        """Plot comparison of different energy saving strategies"""
        strategies = [
            'Default Mode',
            'Reduce Brightness 50%',
            'Close Background Apps',
            'Use WiFi instead of 5G',
            'Enable Power Saving Mode',
            'Optimize Screen Timeout',
            'Disable GPS',
            'Comprehensive Optimization'
        ]

        # Base endurance time (hours)
        base_endurance = 12.0

        # Energy saving effect of each strategy
        energy_saving_effects = {
            'Default Mode': 0,
            'Reduce Brightness 50%': 25,
            'Close Background Apps': 15,
            'Use WiFi instead of 5G': 20,
            'Enable Power Saving Mode': 30,
            'Optimize Screen Timeout': 10,
            'Disable GPS': 8,
            'Comprehensive Optimization': 50
        }

        # Calculate endurance time for each strategy
        endurance_times = []
        for strategy in strategies:
            improvement = energy_saving_effects[strategy]
            endurance = base_endurance * (1 + improvement / 100)
            endurance_times.append(endurance)

        # Plot horizontal bar chart
        y_pos = np.arange(len(strategies))
        colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(strategies)))

        bars = ax.barh(y_pos, endurance_times, color=colors, alpha=0.8, edgecolor='black')

        # Add value labels
        for i, (bar, time) in enumerate(zip(bars, endurance_times)):
            improvement = energy_saving_effects[strategies[i]]
            ax.text(time + 0.1, bar.get_y() + bar.get_height() / 2,
                    f'{time:.1f}h (+{improvement}%)',
                    va='center', fontsize=9, fontweight='bold')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(strategies, fontsize=11)
        ax.set_xlabel('Endurance Time (hours)', fontsize=12)
        ax.set_title('Energy Saving Strategies Comparison', fontsize=14, fontweight='bold', y=1.15)
        ax.grid(True, alpha=0.3, axis='x')

        # Mark base endurance line
        ax.axvline(x=base_endurance, color='red', linestyle='--', alpha=0.7,
                   label=f'Base Endurance: {base_endurance}h')
        ax.legend(loc='lower right')

        # Add summary
        max_improvement = max(energy_saving_effects.values())
        best_strategy = max(energy_saving_effects, key=energy_saving_effects.get)
        ax.text(0.02, 0.98, f'Best Strategy: {best_strategy}\nImprovement: {max_improvement}%',
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

# ==================== 主程序 ====================
def main():
    """主函数：运行电池模型并生成可视化"""
    print("智能手机电池模型与可视化系统")

    # 创建可视化器
    visualizer = BatteryVisualizer()

    # 创建用户行为模型
    user_model = UserBehaviorModel(seed=42)

    # 参数设置
    aging_alpha = 0.0015

    # 运行静态场景
    print("运行静态场景模拟...")
    static_results = {}

    for scenario in ['light', 'typical', 'heavy']:
        params = {
            'scenario': scenario,
            'cycle_count': 1000,
            'aging_alpha': aging_alpha,
            'network_config': user_model.scenario_configs[scenario]
        }

        sol_static = simulate_battery_drain(
            params, initial_SOC=100, t_end=48 * 3600, dynamic_mode=False
        )

        if sol_static is not None and hasattr(sol_static, 't'):
            static_results[scenario] = {
                'SOC': sol_static.y[0] if sol_static.y.ndim > 1 else sol_static.y,
                't': sol_static.t
            }
            print(f"{scenario}场景模拟完成")

    # 运行动态场景
    print("运行动态场景模拟...")
    params_dynamic = {
        'cycle_count': 1000,
        'aging_alpha': aging_alpha,
    }

    sol_dynamic = simulate_battery_drain(
        params_dynamic, initial_SOC=100, t_end=24 * 3600,
        dynamic_mode=True, user_model=user_model
    )

    # 计算续航时间
    if sol_dynamic is not None and hasattr(sol_dynamic, 't_events'):
        if len(sol_dynamic.t_events) > 0 and sol_dynamic.t_events[0].size > 0:
            time_to_empty = sol_dynamic.t_events[0][0] / 3600
        else:
            time_to_empty = sol_dynamic.t[-1] / 3600
    else:
        time_to_empty = 24

    print(f"动态场景续航时间: {time_to_empty:.2f} 小时")

    # 生成独立的可视化图表
    print("生成独立的可视化图表...")
    charts = visualizer.create_individual_charts(sol_dynamic, static_results,
                                                 user_model, time_to_empty)

    print("所有独立图表生成完成！")

if __name__ == "__main__":
    np.random.seed(42)
    main()