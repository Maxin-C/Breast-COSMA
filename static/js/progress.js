document.addEventListener('DOMContentLoaded', function () {

    const followupListContainer = document.getElementById('followup-list');
    const exerciseListContainer = document.getElementById('exercise-list');
    const followupLoading = document.getElementById('followup-loading');
    const exerciseLoading = document.getElementById('exercise-loading');

    /**
     * 根据周数获取在进度条上的百分比位置
     * @param {number} week - 周数 (0, 2, 4, 6)
     */
    function getWeekPercentage(week) {
        // ✨ 更新点 2: 进度条分割为 7 份
        const totalSegments = 7;
        const percentage = (week / totalSegments) * 100;
        
        // 确保百分比在 0 到 100 之间
        return Math.min(100, Math.max(0, percentage));
    }

    /**
     * 渲染随访进程列表
     * @param {HTMLElement} container - 容器元素
     * @param {Array} patients - 患者数据
     */
    function renderFollowup(container, patients) {
        if (followupLoading) followupLoading.remove();
        container.innerHTML = ''; // 清空

        // ... 省略了其他逻辑 ...

        patients.forEach(patient => {
            const statusClass = {
                '等待随访': 'status-waiting',
                '脱落': 'status-dropout',
                '无需随访': 'status-complete',
            }[patient.status] || 'status-complete';

            // node.week 现在将使用新的百分比
            const nodesHtml = patient.nodes.map(node =>
                `<div class="progress-node status-${node.status}" style="left: ${getWeekPercentage(node.week)}%;"></div>`
            ).join('');

            const rowHtml = `
                <div class="patient-row">
                    <span class="patient-name">${patient.name}</span>
                    <div class="progress-bar-container">
                        <div class="progress-bar-track">
                            <div class="progress-bar-fill" style="width: ${patient.progress_percent}%;"></div>
                            ${nodesHtml}
                        </div>
                    </div>
                    <span class="patient-status ${statusClass}">${patient.status}</span>
                </div>
            `;
            container.insertAdjacentHTML('beforeend', rowHtml);
        });
    }

    /**
     * 渲染锻炼次数列表
     * @param {HTMLElement} container - 容器元素
     * @param {Array} patients - 患者数据
     */
    function renderExercise(container, patients) {
        if (exerciseLoading) exerciseLoading.remove();
        container.innerHTML = ''; // 清空

        if (!patients || patients.length === 0) {
            container.innerHTML = '<div class="patient-row"><span class="patient-name">没有符合条件的患者</span></div>';
            return;
        }

        patients.forEach(patient => {
            const ratioClass = patient.ratio_status === 'low' ? 'status-low' : 'status-normal';
            
            const nodesHtml = patient.nodes.map(node =>
                `<div class="day-node status-${node.status}">${node.status !== 'grey' ? node.count : ''}</div>`
            ).join('');

            const rowHtml = `
                <div class="patient-row exercise-row">
                    <span class="patient-name">${patient.name}</span>
                    <div class="exercise-days-wrapper">
                        <div class="exercise-days-container">
                            ${nodesHtml}
                        </div>
                    </div>
                    <span class="patient-ratio ${ratioClass}">${patient.ratio_percent}</span>
                </div>
            `;
            container.insertAdjacentHTML('beforeend', rowHtml);
        });
    }

    // --- 主函数：获取数据并渲染 ---
    function loadProgressData() {
        fetch('/api/progress_data')
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                if (data.error) {
                    throw new Error(data.error);
                }
                renderFollowup(followupListContainer, data.followup_progress);
                renderExercise(exerciseListContainer, data.exercise_progress);
            })
            .catch(error => {
                console.error('Error fetching progress data:', error);
                if (followupLoading) followupLoading.innerHTML = `<span class="patient-name text-danger">数据加载失败: ${error.message}</span>`;
                if (exerciseLoading) exerciseLoading.innerHTML = `<span class="patient-name text-danger">数据加载失败: ${error.message}</span>`;
            });
    }

    loadProgressData();
});