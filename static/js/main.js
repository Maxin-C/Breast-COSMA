document.addEventListener('DOMContentLoaded', function () {
    const computedStyle = getComputedStyle(document.documentElement);
    const PIE_COLOR_1 = computedStyle.getPropertyValue('--pie-color-1').trim();
    const PIE_COLOR_2 = computedStyle.getPropertyValue('--pie-color-2').trim();
    const PIE_COLOR_3 = computedStyle.getPropertyValue('--pie-color-3').trim();
    const PIE_COLOR_4 = computedStyle.getPropertyValue('--pie-color-4').trim();
    const LINE_COLOR = computedStyle.getPropertyValue('--line-color').trim();
    // 初始化所有图表实例
    const populationChart = echarts.init(document.getElementById('population-chart'));
    const evaluationChart = echarts.init(document.getElementById('evaluation-chart'));
    const qualityChart = echarts.init(document.getElementById('quality-chart'));
    const usageChart = echarts.init(document.getElementById('usage-chart'));

    // 从API获取数据并更新仪表盘
    function updateDashboard() {
        fetch('/api/dashboard/stats')
            .then(response => response.json())
            .then(data => {
                console.log(data.quality_of_life)
                // 1. 更新人群分布图
                const populationData = [
                    { name: '已拔管', value: data.population_distribution['已拔管'] || 0 },
                    { name: '未拔管', value: data.population_distribution['未拔管'] || 0 }

                ];
                const populationOption = {
                    color: [PIE_COLOR_2, PIE_COLOR_1], 
                    tooltip: {
                        trigger: 'item',
                        formatter: '{b}: {c} ({d}%)'
                    },
                    legend: {
                        orient: 'vertical',
                        right: '5%',
                        top: 'center',
                        data: ['未拔管', '已拔管']
                    },
                    series: [{
                        name: '人群分布',
                        type: 'pie',
                        radius: ['50%', '70%'],
                        center: ['40%', '50%'],
                        avoidLabelOverlap: false,
                        label: {
                            show: false,
                            position: 'center'
                        },
                        emphasis: {
                            label: {
                                show: true,
                                fontSize: '20',
                                fontWeight: 'bold'
                            }
                        },
                        labelLine: {
                            show: false
                        },
                        data: populationData
                    }]
                };
                populationChart.setOption(populationOption);

                // 2. 更新评估结果图
                const evaluationCategories = Object.keys(data.evaluation_results);
                const evaluationData = evaluationCategories.map(category => ({
                    name: category,
                    value: data.evaluation_results[category] || 0
                }));
                const evaluationOption = {
                    color: [PIE_COLOR_1, PIE_COLOR_2, PIE_COLOR_3, PIE_COLOR_4],
                    tooltip: {
                        trigger: 'item',
                        formatter: '{b}: {c} ({d}%)'
                    },
                    legend: {
                        orient: 'vertical', right: '5%', top: 'center',
                        data: evaluationCategories
                    },
                    series: [{
                        name: '评估结果', type: 'pie', radius: '70%',
                        center: ['40%', '50%'],
                        data: evaluationData
                    }]
                };
                evaluationChart.setOption(evaluationOption);

                // 3. 更新生活质量图
                const qolCategories = Object.keys(data.quality_of_life);
                const qolData = qolCategories.map(category => ({
                    name: category,
                    value: data.quality_of_life[category] || 0
                }));
                const qualityOption = {
                    color: [PIE_COLOR_1, PIE_COLOR_2, PIE_COLOR_3, PIE_COLOR_4],
                    tooltip: {
                        trigger: 'item',
                        formatter: '{b}: {c} ({d}%)'
                    },
                    legend: {
                        orient: 'vertical', right: '5%', top: 'center',
                        data: qolCategories
                    },
                    series: [{
                        name: '生活质量', type: 'pie', radius: '70%',
                        center: ['40%', '50%'],
                        data: qolData
                    }]
                };
                qualityChart.setOption(qualityOption);
                
                // 4. 更新小程序使用情况图
                const usageDates = data.miniprogram_usage.map(item => item.date.substring(5)); // M-D
                const usageCounts = data.miniprogram_usage.map(item => item.count);
                const usageOption = {
                    color: [LINE_COLOR],
                    tooltip: {
                        trigger: 'axis'
                    },
                    grid: {
                        left: '3%',
                        right: '4%',
                        bottom: '3%',
                        containLabel: true
                    },
                    xAxis: { data: usageDates },
                    series: [{
                        name: '使用次数',
                        type: 'line',
                        smooth: true,
                        data: usageCounts
                    }],
                    yAxis: {
                        type: 'value'
                    }
                };
                usageChart.setOption(usageOption);

                // 5. 更新项目整体情况
                const stats = data.project_stats;
                document.getElementById('total-cases').textContent = stats.total_cases;
                document.getElementById('total-videos').textContent = stats.total_videos;
                document.getElementById('total-reports').textContent = stats.total_reports;
                document.getElementById('duration-days').innerHTML = `${stats.duration_days}<span class="unit">天</span>`;
            })
            .catch(error => console.error('Error fetching dashboard data:', error));
    }
    
    // 首次加载时更新
    updateDashboard();

    // 确保图表在窗口大小变化时自适应
    window.addEventListener('resize', () => {
        populationChart.resize();
        evaluationChart.resize();
        qualityChart.resize();
        usageChart.resize();
    });
});