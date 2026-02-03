import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List

# 导入原模型函数
from model_with_visualization import simulate_battery_drain

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class BatterySensitivityAnalyzer:
    """
    电池模型灵敏度分析器
    分析不同参数变化对续航时间的影响
    """

    def __init__(self, base_params: Dict):
        """
        初始化灵敏度分析器

        Parameters:
        -----------
        base_params : Dict
            基础参数配置
        """
        self.base_params = base_params.copy()
        self.results = {}

    def analyze_parameter_sensitivity(self,
                                      param_name: str,
                                      param_range: List[float],
                                      scenario: str = 'typical',
                                      initial_SOC: float = 100,
                                      t_end: float = 24 * 3600) -> Dict:
        """
        分析单个参数的灵敏度

        Parameters:
        -----------
        param_name : str
            参数名称
        param_range : List[float]
            参数取值范围
        scenario : str
            使用场景
        initial_SOC : float
            初始SOC
        t_end : float
            模拟结束时间

        Returns:
        --------
        Dict : 分析结果
        """
        print(f"正在进行参数灵敏度分析: {param_name}")
        print(f"参数范围: {param_range}")

        results = {
            'param_name': param_name,
            'param_values': [],
            'time_to_empty': [],
            'final_SOC': [],
            'sensitivity_scores': []
        }

        for value in param_range:
            # 更新参数
            params = self.base_params.copy()

            # 特殊处理嵌套参数
            if '.' in param_name:
                parts = param_name.split('.')
                current = params
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            else:
                params[param_name] = value

            # 确保场景参数
            params['scenario'] = scenario

            # 运行模拟
            sol = simulate_battery_drain(
                params,
                initial_SOC=initial_SOC,
                t_end=t_end,
                dynamic_mode=False
            )

            if sol is not None and hasattr(sol, 't'):
                soc_array = sol.y[0] if sol.y.ndim > 1 else sol.y

                # 找到SOC降至1%的时间
                below_threshold = np.where(soc_array <= 1)[0]
                if len(below_threshold) > 0:
                    time_to_empty = sol.t[below_threshold[0]] / 3600
                else:
                    time_to_empty = t_end / 3600

                final_SOC = soc_array[-1] if len(soc_array) > 0 else 0

                results['param_values'].append(value)
                results['time_to_empty'].append(time_to_empty)
                results['final_SOC'].append(final_SOC)

                print(f"{param_name}={value:.4f} → 续航: {time_to_empty:.2f}小时")

        # 计算灵敏度得分
        if len(results['time_to_empty']) > 1:
            # 去除NaN值
            valid_indices = ~np.isnan(results['time_to_empty'])
            if np.sum(valid_indices) >= 2:
                valid_values = np.array(results['param_values'])[valid_indices]
                valid_times = np.array(results['time_to_empty'])[valid_indices]

                # 计算灵敏度
                param_changes = np.diff(valid_values) / (valid_values[:-1] + 1e-10)
                time_changes = np.diff(valid_times) / valid_times[:-1]

                sensitivities = time_changes / param_changes
                results['sensitivity_scores'] = list(sensitivities)

        self.results[param_name] = results
        return results

    def analyze_multiple_parameters(self,
                                    parameters: Dict[str, List[float]],
                                    scenario: str = 'typical',
                                    initial_SOC: float = 100) -> pd.DataFrame:
        """
        分析多个参数的灵敏度

        Parameters:
        -----------
        parameters : Dict[str, List[float]]
            参数名称到取值范围的映射
        scenario : str
            使用场景

        Returns:
        --------
        pd.DataFrame : 汇总结果
        """
        print("多参数灵敏度分析开始")

        all_results = []

        for param_name, param_range in parameters.items():
            result = self.analyze_parameter_sensitivity(
                param_name, param_range, scenario, initial_SOC
            )

            # 计算平均灵敏度
            if result['sensitivity_scores']:
                avg_sensitivity = np.mean(np.abs(result['sensitivity_scores']))
            else:
                avg_sensitivity = np.nan

            # 计算续航变化范围
            valid_times = [t for t in result['time_to_empty'] if not np.isnan(t)]
            if valid_times:
                time_range = max(valid_times) - min(valid_times)
                time_change_percent = (time_range / np.mean(valid_times)) * 100
            else:
                time_range = np.nan
                time_change_percent = np.nan

            all_results.append({
                'Parameter': param_name,
                'Avg_Sensitivity': avg_sensitivity,
                'Time_Range_Hours': time_range,
                'Time_Change_Percent': time_change_percent,
                'Min_Time': min(valid_times) if valid_times else np.nan,
                'Max_Time': max(valid_times) if valid_times else np.nan
            })

        # 创建结果DataFrame
        df_results = pd.DataFrame(all_results)

        # 按灵敏度排序
        df_results = df_results.sort_values('Avg_Sensitivity', ascending=False)

        print("多参数灵敏度分析结果汇总")
        print(df_results.to_string())

        return df_results


    def analyze_temperature_effects(self,
                                    temp_range: List[float] = None,
                                    initial_SOC: float = 100,
                                    t_end: float = 24 * 3600):
        """专门分析温度对电池性能的影响"""
        if temp_range is None:
            temp_range = np.linspace(-10, 50, 13)

        print("温度对电池性能影响分析")

        temperatures = []
        times_to_empty = []
        scenarios_list = []

        for temp in temp_range:
            print(f"分析温度: {temp}°C")

            # 对每个场景进行模拟
            for scenario in ['light', 'typical', 'heavy']:
                params = self.base_params.copy()
                params['scenario'] = scenario

                if scenario == 'light':
                    params['network_config'] = {
                        '4G': {'enabled': False, 'signal_strength': 'good'},
                        '5G': {'enabled': False},
                        'WiFi': {'enabled': False},
                        'Bluetooth': {'enabled': False}
                    }
                elif scenario == 'heavy':
                    params['network_config'] = {
                        '4G': {'enabled': True, 'signal_strength': 'good'},
                        '5G': {'enabled': True, 'signal_strength': 'good'},
                        'WiFi': {'enabled': True, 'signal_strength': 'good'},
                        'Bluetooth': {'enabled': True, 'signal_strength': 'good'}
                    }
                else:
                    params['network_config'] = {
                        '4G': {'enabled': True, 'signal_strength': 'good'},
                        '5G': {'enabled': False},
                        'WiFi': {'enabled': True, 'signal_strength': 'good'},
                        'Bluetooth': {'enabled': False}
                    }

                # 关键修改：传递环境温度参数
                sol = simulate_battery_drain(
                    params,
                    initial_SOC=initial_SOC,
                    t_end=t_end,
                    dynamic_mode=False,
                    ambient_temperature=temp
                )

                if sol is not None and hasattr(sol, 't'):
                    soc_array = sol.y[0] if sol.y.ndim > 1 else sol.y
                    below_threshold = np.where(soc_array <= 1)[0]

                    if len(below_threshold) > 0:
                        time_to_empty = sol.t[below_threshold[0]] / 3600
                    else:
                        time_to_empty = t_end / 3600

                    temperatures.append(temp)
                    times_to_empty.append(time_to_empty)
                    scenarios_list.append(scenario)

                    print(f"{scenario}: 续航 {time_to_empty:.2f} 小时")

        # 创建结果DataFrame
        df_temp = pd.DataFrame({
            'Temperature': temperatures,
            'Time_to_Empty': times_to_empty,
            'Scenario': scenarios_list
        })

        # 绘制温度影响图
        plt.figure(figsize=(12, 8))

        for scenario in ['light', 'typical', 'heavy']:
            df_scenario = df_temp[df_temp['Scenario'] == scenario]
            if len(df_scenario) > 0:
                plt.plot(df_scenario['Temperature'], df_scenario['Time_to_Empty'],
                         marker='o', linewidth=2, label=scenario, markersize=8)

        plt.xlabel('Environment Temperature (°C)', fontsize=12)  # 改为英文
        plt.ylabel('Endurance Time (hours)', fontsize=12)  # 改为英文
        plt.title('Impact of Temperature on Battery Endurance', fontsize=14, fontweight='bold')  # 改为英文
        plt.grid(True, alpha=0.3)
        plt.legend()

        # 标记最佳温度范围
        plt.axvspan(15, 30, alpha=0.2, color='green', label='Optimal Temperature Range')
        plt.legend()

        plt.tight_layout()
        plt.show()

        return df_temp

    def analyze_network_config_sensitivity(self,
                                           initial_SOC: float = 100):
        """
        专门分析网络配置对电池续航的影响
        """
        print("网络配置灵敏度分析")

        network_configs = [
            {
                'name': '仅4G',
                'config': {
                    '4G': {'enabled': True, 'signal_strength': 'good'},
                    '5G': {'enabled': False},
                    'WiFi': {'enabled': False},
                    'Bluetooth': {'enabled': False}
                }
            },
            {
                'name': '4G+WiFi',
                'config': {
                    '4G': {'enabled': True, 'signal_strength': 'good'},
                    '5G': {'enabled': False},
                    'WiFi': {'enabled': True, 'signal_strength': 'good'},
                    'Bluetooth': {'enabled': False}
                }
            },
            {
                'name': '5G+WiFi',
                'config': {
                    '4G': {'enabled': False, 'signal_strength': 'good'},
                    '5G': {'enabled': True, 'signal_strength': 'good'},
                    'WiFi': {'enabled': True, 'signal_strength': 'good'},
                    'Bluetooth': {'enabled': False}
                }
            },
            {
                'name': '全开',
                'config': {
                    '4G': {'enabled': True, 'signal_strength': 'good'},
                    '5G': {'enabled': True, 'signal_strength': 'good'},
                    'WiFi': {'enabled': True, 'signal_strength': 'good'},
                    'Bluetooth': {'enabled': True, 'signal_strength': 'good'}
                }
            },
            {
                'name': '飞行模式',
                'config': {
                    '4G': {'enabled': False, 'signal_strength': 'good'},
                    '5G': {'enabled': False},
                    'WiFi': {'enabled': False},
                    'Bluetooth': {'enabled': False}
                }
            }
        ]

        results = []

        for config_info in network_configs:
            config_name = config_info['name']
            network_config = config_info['config']

            print(f"测试配置: {config_name}")

            params = self.base_params.copy()
            params['scenario'] = 'typical'
            params['network_config'] = network_config

            sol = simulate_battery_drain(
                params,
                initial_SOC=initial_SOC,
                t_end=24 * 3600,
                dynamic_mode=False
            )

            if sol is not None and hasattr(sol, 't'):
                soc_array = sol.y[0] if sol.y.ndim > 1 else sol.y
                below_threshold = np.where(soc_array <= 1)[0]

                if len(below_threshold) > 0:
                    time_to_empty = sol.t[below_threshold[0]] / 3600
                else:
                    time_to_empty = 24

                results.append({
                    'Network_Config': config_name,
                    'Time_to_Empty': time_to_empty,
                    'Power_Saving_Percent': (results[0]['Time_to_Empty'] - time_to_empty) / results[0][
                        'Time_to_Empty'] * 100
                    if results else 0
                })

                print(f"续航: {time_to_empty:.2f} 小时")

        # 绘制结果
        df_network = pd.DataFrame(results)

        plt.figure(figsize=(10, 6))
        bars = plt.bar(df_network['Network_Config'], df_network['Time_to_Empty'],
                       color='skyblue')

        plt.xlabel('Network Configuration', fontsize=12)
        plt.ylabel('Endurance Time (hours)', fontsize=12)
        plt.title('Impact of Network Configuration on Battery Endurance', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height + 0.1,
                     f'{height:.2f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

        return df_network

    def plot_sensitivity_radar(self, df_results, save_path=None):
        """
        绘制参数敏感性分析雷达图

        Parameters:
        -----------
        df_results : pd.DataFrame
            灵敏度分析结果DataFrame
        save_path : str
            保存路径
        """
        if df_results.empty:
            print("没有可用的灵敏度数据")
            return

        # 选择前6个参数用于雷达图
        n_params = min(6, len(df_results))
        top_params = df_results.head(n_params)

        # 获取参数名称和灵敏度值
        param_names = top_params['Parameter'].tolist()

        # 准备多个指标数据
        metrics = {
            'sensitivity_score': top_params['Avg_Sensitivity'].fillna(0).tolist(),
            'endurance_change_percent(%)': top_params['Time_Change_Percent'].fillna(0).tolist(),
            'endurance_range_hours': top_params['Time_Range_Hours'].fillna(0).tolist()
        }

        # 创建雷达图
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(projection='polar'))

        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        metric_names = list(metrics.keys())

        for idx, (metric_name, metric_data) in enumerate(metrics.items()):
            ax = axes[idx]

            # 准备雷达图数据
            angles = np.linspace(0, 2 * np.pi, n_params, endpoint=False).tolist()
            values = metric_data

            # 归一化到0-1范围
            if max(values) > 0:
                values_normalized = np.array(values) / max(values)
            else:
                values_normalized = np.zeros_like(values)

            # 闭合曲线
            values_normalized = np.concatenate((values_normalized, [values_normalized[0]]))
            angles += angles[:1]

            # 绘制雷达图
            ax.plot(angles, values_normalized, 'o-', linewidth=2, color=colors[idx],
                    label=metric_name)
            ax.fill(angles, values_normalized, alpha=0.25, color=colors[idx])

            # 设置角度标签
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(param_names, fontsize=10)

            # 设置径向标签
            ax.set_ylim(0, 1.2)
            ax.set_yticks([0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=9)

            # 添加标题
            ax.set_title(f'{metric_name} Radar Chart', fontsize=12, fontweight='bold', pad=20)

            # 添加数值标签
            for i, (angle, value, orig_value) in enumerate(zip(angles[:-1], values_normalized[:-1], values)):
                if metric_name == '灵敏度得分':
                    value_str = f'{orig_value:.3f}'
                elif metric_name == '续航变化幅度(%)':
                    value_str = f'{orig_value:.1f}%'
                else:
                    value_str = f'{orig_value:.1f}h'

                label_angle = angle
                label_radius = value + 0.1

                ax.text(label_angle, label_radius, value_str,
                        ha='center', va='center', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        plt.suptitle('Parameter Sensitivity Analysis Radar Chart', fontsize=16, fontweight='bold', y=1.05)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"雷达图已保存到: {save_path}")

        plt.show()

        # 创建一个综合雷达图
        self._plot_comprehensive_radar(top_params, param_names)

    def _plot_comprehensive_radar(self, top_params, param_names):
        """绘制综合雷达图，显示所有指标"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='polar')

        n_params = len(param_names)
        angles = np.linspace(0, 2 * np.pi, n_params, endpoint=False).tolist()

        # 准备多个指标数据
        metrics_data = {
            'sensitivity': top_params['Avg_Sensitivity'].fillna(0).tolist(),
            'endurance_change_%': top_params['Time_Change_Percent'].fillna(0).tolist(),
            'endurance_range': top_params['Time_Range_Hours'].fillna(0).tolist(),
            'min_endurance': top_params['Min_Time'].fillna(0).tolist(),
            'max_endurance': top_params['Max_Time'].fillna(0).tolist()
        }

        # 归一化每个指标
        normalized_metrics = {}
        for metric_name, values in metrics_data.items():
            if max(values) > 0:
                normalized = np.array(values) / max(values)
            else:
                normalized = np.zeros_like(values)
            normalized_metrics[metric_name] = normalized

        # 绘制每个指标的雷达图
        colors = plt.cm.Set2(np.linspace(0, 1, len(metrics_data)))

        for (metric_name, values), color in zip(normalized_metrics.items(), colors):
            values_closed = np.concatenate((values, [values[0]]))
            angles_closed = angles + angles[:1]

            ax.plot(angles_closed, values_closed, 'o-', linewidth=2,
                    color=color, label=metric_name, markersize=6)
            ax.fill(angles_closed, values_closed, alpha=0.1, color=color)

        # 设置角度标签
        ax.set_xticks(angles)
        ax.set_xticklabels(param_names, fontsize=11)

        # 设置径向标签
        ax.set_ylim(0, 1.2)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=9)

        # 添加标题和图例
        ax.set_title('Comprehensive Sensitivity Radar Chart\n(Normalized Comparison of All Metrics)', fontsize=14,
                     fontweight='bold', pad=30)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)

        # 添加参数重要性说明
        most_sensitive = param_names[0]
        least_sensitive = param_names[-1]

        info_text = f"Most Sensitive Parameter: {most_sensitive}\n"
        info_text += f"Least Sensitive Parameter: {least_sensitive}"

        ax.text(0.5, -0.15, info_text, transform=ax.transAxes, ha='center',
                fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        plt.tight_layout()
        plt.savefig('sensitivity_comprehensive_radar.png', dpi=300, bbox_inches='tight')
        plt.show()


def run_comprehensive_sensitivity_analysis():
    """
    运行完整的灵敏度分析
    """
    print("智能手机电池模型综合灵敏度分析")

    # 基础参数
    base_params = {
        'scenario': 'typical',
        'cycle_count': 500,
        'aging_alpha': 0.0015,
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

    # 创建分析器
    analyzer = BatterySensitivityAnalyzer(base_params)

    # 定义要分析的参数范围
    parameters_to_analyze = {
        'base_power': np.linspace(0.3, 1.0, 8),
        'n_background_apps': np.arange(0, 10, 2),
        'screen_timeout': [60, 120, 300, 600, 1200],
        'aging_alpha': np.linspace(0.0005, 0.003, 7),
        'cycle_count': [0, 200, 500, 1000, 1500, 2000],
        'gps_enabled': [0, 1]
    }

    # 运行多参数分析
    df_results = analyzer.analyze_multiple_parameters(
        parameters=parameters_to_analyze,
        scenario='typical',
        initial_SOC=100
    )

    # 温度影响分析
    df_temp = analyzer.analyze_temperature_effects(
        temp_range=np.linspace(-10, 50, 13),
        initial_SOC=100
    )

    # 网络配置分析
    df_network = analyzer.analyze_network_config_sensitivity(
        initial_SOC=100
    )

    # 绘制敏感性分析雷达图
    analyzer.plot_sensitivity_radar(
        df_results=df_results,
        save_path='sensitivity_radar_chart.png'
    )

    return analyzer, df_results, df_temp, df_network


if __name__ == "__main__":
    analyzer, df_results, df_temp, df_network = run_comprehensive_sensitivity_analysis()

    print("灵敏度分析完成")
    print("主要发现:")
    print(f"1. 最敏感的参数: {df_results.iloc[0]['Parameter']}")
    print(f"2. 最不敏感的参数: {df_results.iloc[-1]['Parameter']}")
    print(f"3. 网络配置对续航的影响范围: {df_network['Time_to_Empty'].max() - df_network['Time_to_Empty'].min():.2f} 小时")