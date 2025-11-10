from flask import Blueprint, render_template, jsonify, request
from sqlalchemy import func, case
from datetime import datetime, timedelta, date, time

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

@main_bp.route('/progress')
@login_required
def progress_page():
    return render_template('progress.html', active_page='progress')

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
                (RecoveryRecordDetail.evaluation_details == None, '未评估'),
                (RecoveryRecordDetail.evaluation_details == '', '未评估'),
                else_='已评估'
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

@main_bp.route('/api/progress_data', methods=['GET'])
@login_required
def get_progress_data():
    """
    为随访进度页提供所有需要计算的数据。
    """
    try:
        today = date.today()
        # 您的要求是 2025年9月15日 之后注册的患者
        REGISTRATION_START_DATE = date(2025, 9, 15) 
        FOLLOWUP_WEEKS = [0, 2, 4, 6] # 第0, 2, 4, 6周
        EXERCISE_DAYS_TOTAL = 42 # 共记录42天

        # 1. 获取所有符合条件的患者
        users = User.query.filter(
            User.registration_date >= REGISTRATION_START_DATE
        ).order_by(User.name).all()

        followup_data = []
        exercise_data = []

        for user in users:
            if not user.registration_date:
                continue
            
            registration_date = user.registration_date.date()
            days_since_registration = (today - registration_date).days
            
            if days_since_registration < 0: # 排除尚未注册的（以防万一）
                continue

            # --- 2. 计算随访进程 (Follow-up Progress) ---
            followup_nodes = []
            has_yellow = False
            has_red = False
            
            # 计算总进度条的百分比 (从注册日到第6周结束)
            max_followup_week_days = (FOLLOWUP_WEEKS[-1] * 7) + 6 # 第6周(day 42)到第48天
            progress_percentage = (days_since_registration / max_followup_week_days) * 100
            
            all_reachable_nodes_are_green = True # 假设所有已到达的节点都是绿色

            for week in FOLLOWUP_WEEKS:
                node_status = "grey" # 默认灰色 (未来)
                week_start_day = week * 7
                week_end_day = week_start_day + 6 # 第0周是 0-6 天

                # 检查是否到达或超过了这一周的开始
                if days_since_registration >= week_start_day:
                    # 这一周是当前周或过去周
                    week_start_dt = datetime.combine(registration_date + timedelta(days=week_start_day), time.min)
                    week_end_dt = datetime.combine(registration_date + timedelta(days=week_end_day), time.max)
                    
                    # 查询该周内是否有QoL记录
                    record_exists = db.session.query(QoL.qol_id).filter(
                        QoL.user_id == user.user_id,
                        QoL.submission_time.between(week_start_dt, week_end_dt)
                    ).first()

                    if record_exists:
                        node_status = "green"
                    else:
                        # 没有记录，判断是错过了还是正在等
                        if days_since_registration > week_end_day: # 已经过了这一周
                            node_status = "red"
                            has_red = True
                            all_reachable_nodes_are_green = False
                        else: # 正在这一周内
                            node_status = "yellow"
                            has_yellow = True
                            all_reachable_nodes_are_green = False
                else:
                    # 还没到这一周
                    all_reachable_nodes_are_green = False

                followup_nodes.append({"week": week, "status": node_status})

            # 确定总体状态
            overall_status = "无需随访"
            if has_red:
                overall_status = "脱落"
            elif has_yellow:
                overall_status = "等待随访"
            elif all_reachable_nodes_are_green and days_since_registration > max_followup_week_days:
                # 所有节点都是绿色，并且已经过了最后期限
                overall_status = "无需随访"


            followup_data.append({
                "user_id": user.user_id,
                "name": user.name,
                "nodes": followup_nodes,
                "status": overall_status,
                "progress_percent": min(progress_percentage, 100)
            })

            # --- 3. 计算锻炼次数 (Exercise Progress) ---
            start_dt = datetime.combine(registration_date, time.min)
            # 42天，即 day 0 到 day 41
            end_dt = datetime.combine(registration_date + timedelta(days=EXERCISE_DAYS_TOTAL), time.min)

            # 一次性查询该用户42天内的所有锻炼记录
            exercise_counts_query = db.session.query(
                func.date(RecoveryRecord.record_date).label('date'),
                func.count(RecoveryRecord.record_id).label('count')
            ).filter(
                RecoveryRecord.user_id == user.user_id,
                RecoveryRecord.record_date >= start_dt,
                RecoveryRecord.record_date < end_dt # 小于第42天的开始 (即包含第0-41天)
            ).group_by(func.date(RecoveryRecord.record_date)).all()
            
            # 转为字典以便快速查找
            counts_dict = {r.date: r.count for r in exercise_counts_query}

            exercise_nodes = []
            green_days = 0
            red_days = 0

            # 遍历 0 到 41 (共42天)
            for day_num in range(EXERCISE_DAYS_TOTAL):
                current_day_date = registration_date + timedelta(days=day_num)
                day_status = "grey" # 默认灰色 (未来)
                count = counts_dict.get(current_day_date, 0)
                
                # *** 修正后的逻辑 ***
                # 检查这一天是今天、过去还是未来
                if today >= current_day_date: 
                    # 如果是今天或过去
                    if count > 0:
                        day_status = "green"
                        green_days += 1
                    else:
                        # 严格按照要求：等于0为红色 (无论是今天还是过去)
                        day_status = "red"
                        red_days += 1
                # else: day_status 保持 "grey" (未来)

                exercise_nodes.append({
                    "day": day_num + 1, # Day 1 到 Day 42
                    "status": day_status,
                    "count": count
                })

            # 计算占比
            total_days_evaluated = green_days + red_days
            ratio = (green_days / total_days_evaluated * 100) if total_days_evaluated > 0 else 0
            ratio_status = "low" if ratio < 50 else "normal"

            exercise_data.append({
                "user_id": user.user_id,
                "name": user.name,
                "nodes": exercise_nodes,
                "ratio_percent": f"{ratio:.0f}%",
                "ratio_status": ratio_status
            })
            
        return jsonify({
            "followup_progress": followup_data,
            "exercise_progress": exercise_data
        })

    except Exception as e:
        print(f"Error in get_progress_data: {e}") # 打印错误到后端日志
        return jsonify({"error": str(e)}), 500