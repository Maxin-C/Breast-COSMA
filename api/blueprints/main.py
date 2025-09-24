from flask import Blueprint, render_template, jsonify, request
from sqlalchemy import func, case
from datetime import datetime, timedelta

from api.extensions import db
from api.decorators import login_required
from utils.database.models import User, RecoveryRecord, RecoveryRecordDetail, QoL, NurseEvaluation

main_bp = Blueprint('main', __name__)

# --- Page Rendering Routes ---
@main_bp.route('/')
@login_required
def index_page():
    return render_template('dashboard.html')

@main_bp.route('/cases')
@login_required
def cases_page():
    return render_template('cases.html')

@main_bp.route('/case/<int:user_id>')
@login_required
def case_detail_page(user_id):
    return render_template('case_detail.html', user_id=user_id)

# --- APIs for New Frontend ---
@main_bp.route('/api/dashboard/stats', methods=['GET'])
@login_required
def get_dashboard_stats():
    try:
        # 1. 人群分布
        pop_dist = db.session.query(
            User.extubation_status,
            func.count(User.user_id)
        ).group_by(User.extubation_status).all()
        population_distribution = {status: count for status, count in pop_dist}

        # 2. 评估结果
        eval_res = db.session.query(
            case(
                (RecoveryRecordDetail.brief_evaluation == None, '未评估'),
                (RecoveryRecordDetail.brief_evaluation == '', '未评估'),
                else_=RecoveryRecordDetail.brief_evaluation
            ).label('evaluation_category'),
            func.count(RecoveryRecordDetail.record_detail_id)
        ).group_by('evaluation_category').all()
        evaluation_results = {category: count for category, count in eval_res}

        # 3. 生活质量
        qol_res = db.session.query(
            # 使用 case 语句将 NULL 或空字符串的 level 映射为 '未评估'
            case(
                (QoL.level == None, '未评估'),
                (QoL.level == '', '未评估'),
                else_=QoL.level
            ).label('qol_category'),
            func.count(QoL.qol_id)
        ).group_by('qol_category').all()
        quality_of_life = {category: count for category, count in qol_res}

        # 4. 项目整体情况
        total_cases = db.session.query(func.count(User.user_id)).scalar()
        total_videos = db.session.query(func.count(RecoveryRecordDetail.record_detail_id)).filter(RecoveryRecordDetail.video_path != None).scalar()
        total_reports = db.session.query(func.count(RecoveryRecordDetail.record_detail_id)).filter(
            RecoveryRecordDetail.evaluation_details != None,
            RecoveryRecordDetail.evaluation_details != ''
        ).scalar()
        
        first_reg = db.session.query(func.min(User.registration_date)).scalar()
        duration_days = (datetime.now() - first_reg).days if first_reg else 0

        # 5. 小程序使用情况
        seven_days_ago = datetime.now() - timedelta(days=7)
        usage_data = db.session.query(
            func.date(RecoveryRecord.record_date),
            func.count(RecoveryRecord.record_id)
        ).filter(
            RecoveryRecord.record_date >= seven_days_ago
        ).group_by(func.date(RecoveryRecord.record_date)).order_by(func.date(RecoveryRecord.record_date)).all()
        miniprogram_usage = [{"date": d.strftime("%Y-%m-%d"), "count": c} for d, c in usage_data]

        return jsonify({
            "population_distribution": population_distribution,
            "evaluation_results": evaluation_results,
            "quality_of_life": quality_of_life,
            "project_stats": {
                "total_cases": total_cases,
                "total_videos": total_videos,
                "total_reports": total_reports,
                "duration_days": duration_days
            },
            "miniprogram_usage": miniprogram_usage
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@main_bp.route('/api/cases_datatable', methods=['GET'])
@login_required
def get_cases_for_datatable():
    """为 DataTables.net 组件提供数据，支持其服务端处理模式。"""
    try:
        # 1. 解析 DataTables 发来的标准请求参数
        draw = request.args.get('draw', 1, type=int)
        start = request.args.get('start', 0, type=int)
        length = request.args.get('length', 10, type=int)
        search_value = request.args.get('search[value]', '', type=str)
        
        # 获取排序列和方向
        order_column_index = request.args.get('order[0][column]', 0, type=int)
        order_column_name = request.args.get(f'columns[{order_column_index}][data]', 'id', type=str)
        order_direction = request.args.get('order[0][dir]', 'asc', type=str)

        # 2. 构建子查询 (与之前版本相同)
        session_count_subq = db.session.query(
            RecoveryRecord.user_id,
            func.count(RecoveryRecord.record_id.distinct()).label('session_count')
        ).join(
            RecoveryRecordDetail, RecoveryRecord.record_id == RecoveryRecordDetail.record_id
        ).filter(
            RecoveryRecordDetail.video_path != None
        ).group_by(
            RecoveryRecord.user_id
        ).subquery()

        # 3. 构建基础查询
        base_query = db.session.query(
            User,
            session_count_subq.c.session_count,
        ).outerjoin(
            session_count_subq, User.user_id == session_count_subq.c.user_id
        )

        # 4. 应用 DataTables 的全局搜索条件
        if search_value:
            base_query = base_query.filter(
                db.or_(
                    User.name.like(f"%{search_value}%"),
                    func.cast(User.srrsh_id, db.String).like(f"%{search_value}%")
                )
            )
        
        # 5. 获取过滤前后的总记录数 (DataTables 要求)
        total_records = db.session.query(func.count(User.user_id)).scalar()
        filtered_records = base_query.count()

        # 6. 应用排序条件
        sort_map = {
            'id': User.user_id,
            'name': User.name,
            'case_id': User.srrsh_id,
            'reg_date': User.registration_date,
            'sessions': session_count_subq.c.session_count
        }
        if order_column_name in sort_map:
            sort_column = sort_map[order_column_name]
            if order_direction == 'desc':
                base_query = base_query.order_by(sort_column.desc())
            else:
                base_query = base_query.order_by(sort_column.asc())

        # 7. 应用分页
        paginated_results = base_query.offset(start).limit(length).all()
        
        # 8. 格式化数据
        cases_data = []
        for user, session_count in paginated_results:
            cases_data.append({
                "id": user.user_id,
                "name": user.name,
                "case_id": user.srrsh_id,
                "reg_date": user.registration_date.strftime('%Y.%m.%d') if user.registration_date else None,
                "category": user.extubation_status,
                "sessions": session_count or 0,
                "result": "无评估", # 简化：如需最新评估，需加入之前的latest_detail_subq
                "user_id": user.user_id # 用于生成链接
            })

        # 9. 按照 DataTables 要求的格式返回 JSON
        return jsonify({
            "draw": draw,
            "recordsTotal": total_records,
            "recordsFiltered": filtered_records,
            "data": cases_data
        })

    except Exception as e:
        print(f"Error in get_cases_for_datatable: {e}")
        return jsonify({"error": str(e)}), 500

@main_bp.route('/api/case/<int:user_id>', methods=['GET'])
@login_required
def get_case_details_api(user_id):
    try:
        user = User.query.get(user_id)
        if not user:
            return jsonify({"error": "User not found"}), 404

        # 1. 获取该用户所有有效的、已生成视频的康复记录详情
        details = db.session.query(
            RecoveryRecord.record_id,
            RecoveryRecord.record_date,
            RecoveryRecordDetail.record_detail_id,
            RecoveryRecordDetail.completion_timestamp,
            RecoveryRecordDetail.evaluation_details,
            RecoveryRecordDetail.video_path,
            RecoveryRecordDetail.exercise_id
        ).join(RecoveryRecordDetail, RecoveryRecord.record_id == RecoveryRecordDetail.record_id)\
         .filter(RecoveryRecord.user_id == user_id, RecoveryRecordDetail.video_path != None)\
         .order_by(RecoveryRecord.record_date.desc(), RecoveryRecordDetail.completion_timestamp.asc()).all()

        # 2. 将扁平化的数据重组为嵌套结构
        records_dict = {}
        for detail in details:
            if detail.record_id not in records_dict:
                records_dict[detail.record_id] = {
                    "record_id": detail.record_id,
                    "record_date": detail.record_date.strftime('%Y-%m-%d'),
                    "details": []
                }
            
            records_dict[detail.record_id]["details"].append({
                "record_detail_id": detail.record_detail_id,
                "completion_timestamp": detail.completion_timestamp.strftime('%H:%M:%S'),
                "evaluation_details": detail.evaluation_details or "暂无详细报告。",
                "video_path": detail.video_path,
                "exercise_id": detail.exercise_id
            })

        nested_records = list(records_dict.values())

        return jsonify({
            "name": user.name,
            "case_id": user.srrsh_id,
            "records": nested_records
        })
    except Exception as e:
        print(f"Error in get_case_details_api: {e}")
        return jsonify({"error": "Internal server error"}), 500

@main_bp.route('/api/case/evaluations', methods=['POST'])
@login_required
def submit_nurse_evaluation():
    # ... (Copy the entire submit_nurse_evaluation function here)
    data = request.json
    if not data or 'record_detail_id' not in data or 'score' not in data:
        return jsonify({"error": "缺少必要参数"}), 400
    try:
        new_evaluation = NurseEvaluation(
            record_detail_id=data['record_detail_id'],
            nurse_id=session['nurse_id'],
            score=data['score'],
            feedback_text=data.get('feedback_text', ''),
            evaluation_timestamp=datetime.now()
        )
        db.session.add(new_evaluation)
        db.session.commit()
        return jsonify({"message": "评估提交成功！"}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"数据库错误: {str(e)}"}), 500