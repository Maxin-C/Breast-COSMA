// static/js/case_detail.js (Completely New Version)
document.addEventListener('DOMContentLoaded', function () {
    // 获取主要DOM元素
    const patientInfoContainer = document.querySelector('.patient-info');
    const accordionContainer = document.getElementById('recordsAccordion');
    const reportContentBox = document.getElementById('report-content');
    const videoPlayer = document.querySelector('.video-player-wrapper video');
    const evaluationForm = document.getElementById('evaluation-form');
    // ... 其他元素获取

    const userId = document.body.dataset.userId;

    /**
     * 主函数：获取数据并构建UI
     */
    function initializePage() {
        if (!userId) {
            console.error("User ID not found.");
            return;
        }

        fetch(`/api/case/${userId}`)
            .then(response => response.json())
            .then(data => {
                // 1. 填充患者信息
                patientInfoContainer.innerHTML = `
                    <h5 class="mb-1">姓名: ${data.name}</h5>
                    <p class="text-muted mb-0">病例号: ${data.case_id}</p>
                `;

                // 2. 动态构建二级折叠菜单
                buildAccordionMenu(data.records);

                // 3. 初始化事件监听
                setupEventListeners();

                // 4. 默认选中第一个记录的第一个练习
                const firstSubItem = document.querySelector('.sub-list-item');
                if (firstSubItem) {
                    firstSubItem.click();
                } else {
                    reportContentBox.textContent = '该用户暂无康复记录。';
                }
            })
            .catch(error => {
                console.error('Error fetching case details:', error);
            });
    }

    /**
     * 根据API返回的嵌套数据构建Bootstrap Accordion菜单
     * @param {Array} records - 包含康复记录和详情的数组
     */
    function buildAccordionMenu(records) {
        accordionContainer.innerHTML = ''; // 清空容器
        records.forEach((record, index) => {
            const isFirst = index === 0;
            const accordionItem = `
                <div class="accordion-item">
                    <h2 class="accordion-header" id="heading-${record.record_id}">
                        <button class="accordion-button ${isFirst ? '' : 'collapsed'}" type="button" data-bs-toggle="collapse" data-bs-target="#collapse-${record.record_id}" aria-expanded="${isFirst}" aria-controls="collapse-${record.record_id}">
                            Record #${record.record_id} (${record.record_date})
                        </button>
                    </h2>
                    <div id="collapse-${record.record_id}" class="accordion-collapse collapse ${isFirst ? 'show' : ''}" aria-labelledby="heading-${record.record_id}" data-bs-parent="#recordsAccordion">
                        <div class="accordion-body p-0">
                            <ul class="list-group list-group-flush">
                                ${record.details.map(detail => `
                                    <li class="list-group-item sub-list-item" 
                                        data-video-path="${detail.video_path}"
                                        data-report="${detail.evaluation_details}"
                                        data-record-detail-id="${detail.record_detail_id}">
                                        Exercise #${detail.exercise_id} - ${detail.completion_timestamp}
                                    </li>
                                `).join('')}
                            </ul>
                        </div>
                    </div>
                </div>
            `;
            accordionContainer.insertAdjacentHTML('beforeend', accordionItem);
        });
    }

    /**
     * 设置所有事件监听
     */
    function setupEventListeners() {
        // 使用事件委托来处理所有二级菜单项的点击事件
        accordionContainer.addEventListener('click', function(event) {
            const target = event.target;
            // 确保点击的是二级菜单项
            if (target && target.classList.contains('sub-list-item')) {
                // 移除所有二级菜单项的 'active' 状态
                document.querySelectorAll('.sub-list-item').forEach(item => {
                    item.classList.remove('active');
                });
                // 为当前点击的项添加 'active' 状态
                target.classList.add('active');

                // 更新视频和报告内容
                const reportText = target.dataset.report;
                const videoPath = target.dataset.videoPath;
                
                reportContentBox.textContent = reportText;
                videoPlayer.src = `/${videoPath}`;
                videoPlayer.load();
            }
        });

        // 评估表单的提交事件 (逻辑不变)
        if (evaluationForm) {
            evaluationForm.addEventListener('submit', function(event) {
                event.preventDefault();
                evaluationStatus.textContent = ''; // 清空状态消息

                const activeItem = document.querySelector('.video-list-item.active');
                if (!activeItem) {
                    evaluationStatus.textContent = '错误：请先在左侧选择一个评估记录。';
                    evaluationStatus.className = 'mt-2 text-danger';
                    return;
                }
                
                const recordDetailId = activeItem.dataset.recordDetailId;
                const checkedRating = document.querySelector('.rating-item input[type="radio"]:checked');
                const feedbackText = document.getElementById('feedback-text').value;

                if (!checkedRating) {
                    evaluationStatus.textContent = '错误：评分是必填项。';
                    evaluationStatus.className = 'mt-2 text-danger';
                    return;
                }
                const score = checkedRating.value;

                // 发送数据到后端API
                fetch('/api/case/evaluations', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        record_detail_id: parseInt(recordDetailId),
                        score: parseInt(score),
                        feedback_text: feedbackText
                    })
                })
                .then(response => response.json().then(data => ({ ok: response.ok, data })))
                .then(({ ok, data }) => {
                    if (ok) {
                        evaluationStatus.textContent = data.message || '提交成功！';
                        evaluationStatus.className = 'mt-2 text-success';
                        evaluationForm.reset(); // 成功后清空表单
                    } else {
                        evaluationStatus.textContent = `错误: ${data.error || '提交失败'}`;
                        evaluationStatus.className = 'mt-2 text-danger';
                    }
                })
                .catch(error => {
                    console.error('Evaluation submission error:', error);
                    evaluationStatus.textContent = '提交时发生网络错误，请稍后重试。';
                    evaluationStatus.className = 'mt-2 text-danger';
                });
            });
        }
    }

    // 页面加载入口
    initializePage();
});