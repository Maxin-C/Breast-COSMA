from utils.database.models import db
from datetime import datetime, date, time

# Helper function to safely get data from form or dict
def get_data_or_none(data, key):
    return data.get(key) if data.get(key) != '' else None

# --- Generic CRUD Functions ---
# These functions are designed to be generic and work with any SQLAlchemy model.

def get_all_records(model):
    """
    Retrieves all records for a given model.
    Args:
        model: The SQLAlchemy model class (e.g., User, RecoveryPlan).
    Returns:
        A list of model instances.
    """
    return model.query.all()

def get_record_by_id(model, record_id):
    """
    Retrieves a single record by its primary key.
    Args:
        model: The SQLAlchemy model class.
        record_id: The primary key of the record.
    Returns:
        The model instance or None if not found.
    """
    return model.query.get(record_id)

def get_records_by_field(model, field_name, field_value):
    """
    Retrieves records for a given model filtered by a specific field and its value.
    Args:
        model: The SQLAlchemy model class.
        field_name: The name of the field to filter by (string).
        field_value: The value to match for the given field.
    Returns:
        A list of model instances matching the criteria.
    """
    try:
        # 使用 getattr 获取模型的属性，然后进行过滤
        # 例如：User.query.filter_by(name=field_value).all()
        if hasattr(model, field_name):
            return model.query.filter(getattr(model, field_name) == field_value).all()
        else:
            print(f"Error: Model {model.__name__} does not have field '{field_name}'")
            return []
    except Exception as e:
        print(f"Error querying records by field: {e}")
        return []

def add_record(model, form_data):
    """
    Adds a new record to the database.
    Args:
        model: The SQLAlchemy model class.
        form_data: A dictionary or form object containing the data for the new record.
    Returns:
        The newly created model instance on success, or None on failure.
    """
    try:
        # Create an instance of the model and populate it with form data
        # Handle specific field types if necessary (e.g., datetime, date, time)
        instance = model()
        for field, value in form_data.items():
            # Special handling for DateTimeField, DateField, TimeField if they are strings
            if hasattr(model, field): # Check if the model has this attribute
                column_type = getattr(model, field).type
                if isinstance(column_type, db.DateTime) and isinstance(value, str):
                    setattr(instance, field, datetime.strptime(value, '%Y-%m-%d %H:%M:%S') if value else None)
                elif isinstance(column_type, db.Date) and isinstance(value, str):
                    setattr(instance, field, datetime.strptime(value, '%Y-%m-%d').date() if value else None)
                elif isinstance(column_type, db.Time) and isinstance(value, str):
                    setattr(instance, field, datetime.strptime(value, '%H:%M:%S').time() if value else None)
                elif value == '': # Treat empty strings from forms as None for Optional fields
                    setattr(instance, field, None)
                else:
                    setattr(instance, field, value)

        db.session.add(instance)
        db.session.commit()
        return instance
    except Exception as e:
        db.session.rollback()
        print(f"Error adding record: {e}")
        return None

def update_record(instance, form_data):
    """
    Updates an existing record in the database.
    Args:
        instance: The existing SQLAlchemy model instance to update.
        form_data: A dictionary or form object containing the updated data.
    Returns:
        The updated model instance on success, or None on failure.
    """
    try:
        for field, value in form_data.items():
            if hasattr(instance, field):
                # Special handling for DateTimeField, DateField, TimeField if they are strings
                column_type = getattr(instance.__class__, field).type
                if isinstance(column_type, db.DateTime) and isinstance(value, str):
                    setattr(instance, field, datetime.strptime(value, '%Y-%m-%d %H:%M:%S') if value else None)
                elif isinstance(column_type, db.Date) and isinstance(value, str):
                    setattr(instance, field, datetime.strptime(value, '%Y-%m-%d').date() if value else None)
                elif isinstance(column_type, db.Time) and isinstance(value, str):
                    setattr(instance, field, datetime.strptime(value, '%H:%M:%S').time() if value else None)
                elif value == '': # Treat empty strings from forms as None for Optional fields
                    setattr(instance, field, None)
                else:
                    setattr(instance, field, value)
        db.session.commit()
        return instance
    except Exception as e:
        db.session.rollback()
        print(f"Error updating record: {e}")
        return None

def delete_record(instance):
    """
    Deletes a record from the database.
    Args:
        instance: The SQLAlchemy model instance to delete.
    Returns:
        True on success, False on failure.
    """
    try:
        db.session.delete(instance)
        db.session.commit()
        return True
    except Exception as e:
        db.session.rollback()
        print(f"Error deleting record: {e}")
        return False
