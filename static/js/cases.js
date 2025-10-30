// static/js/cases.js (DataTables Version)
$(document).ready(function() {
    // 使用jQuery选择器找到我们的表格并初始化DataTables
    $('#cases-table').DataTable({
        // 开启服务端处理模式，所有排序、搜索、分页都将交给后端API处理
        processing: true,
        serverSide: true,

        // 配置API接口地址
        ajax: {
            url: "/api/cases_datatable",
            type: "GET"
        },

        // 定义每一列与API返回的JSON数据中的哪个字段对应
        columns: [
            { data: 'id' },
            { data: 'name' },
            { data: 'case_id' },
            { data: 'reg_date' },
            { data: 'category' },
            { data: 'sessions' },
            { data: 'result', orderable: false }, // '评估结果' 列不可排序
            { 
                data: 'user_id',      // 使用 user_id 来构建链接
                orderable: false,   // '链接' 列不可排序
                searchable: false,  // '链接' 列不可搜索
                render: function(data, type, row) {
                    // 'render' 函数允许我们自定义列的显示内容
                    // 在这里，我们根据 user_id 创建一个 "查看信息" 的链接
                    return `<a href="/case/${data}" class="link-primary">查看信息</a>`;
                }
            }
        ],

        // 默认排序：按第一列（序号）升序排列
        order: [[0, 'asc']],

        // 将DataTables的UI文本设置为中文
        language: {
            "processing": "处理中...",
            "lengthMenu": "显示 _MENU_ 项结果",
            "zeroRecords": "没有匹配结果",
            "info": "显示第 _START_ 至 _END_ 项结果，共 _TOTAL_ 项",
            "infoEmpty": "显示第 0 至 0 项结果，共 0 项",
            "infoFiltered": "(由 _MAX_ 项结果过滤)",
            "search": "搜索（姓名）:", 
            "paginate": {
                "first": "首页",
                "previous": "上页",
                "next": "下页",
                "last": "末页"
            }
        }
    });
});