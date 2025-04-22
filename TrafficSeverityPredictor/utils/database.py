"""
Database utilities for the Traffic Severity Prediction App
"""

import os
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

# Get database URL from environment variable
DATABASE_URL = os.environ.get('DATABASE_URL')
if DATABASE_URL and DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

# Create SQLAlchemy engine and session
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
Base = declarative_base()

class PredictionRecord(Base):
    """Model for storing prediction history"""
    __tablename__ = 'prediction_history'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    day_of_week = Column(String(10), nullable=False)
    hour = Column(Integer, nullable=False)
    weather_condition = Column(String(20), nullable=False)
    road_type = Column(String(20), nullable=False)
    junction_type = Column(String(20), nullable=False)
    traffic_volume = Column(Integer, nullable=False)
    speed_limit = Column(Integer, nullable=False)
    predicted_severity = Column(String(10), nullable=False)
    confidence = Column(Float, nullable=False)

class TrafficIncident(Base):
    """Model for storing traffic incident data"""
    __tablename__ = 'traffic_incidents'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    day_of_week = Column(String(10), nullable=False)
    hour = Column(Integer, nullable=False)
    weather_condition = Column(String(20), nullable=False)
    road_type = Column(String(20), nullable=False)
    junction_type = Column(String(20), nullable=False)
    traffic_volume = Column(Integer, nullable=False)
    speed_limit = Column(Integer, nullable=False)
    severity = Column(String(10), nullable=False)
    severity_numeric = Column(Integer, nullable=False)
    clearance_time = Column(Integer, nullable=False)

def init_db():
    """Initialize the database schema"""
    Base.metadata.create_all(engine)

def save_prediction(prediction_data):
    """
    Save a prediction record to the database
    
    Args:
        prediction_data: Dictionary with prediction data
    """
    session = Session()
    try:
        record = PredictionRecord(
            day_of_week=prediction_data['Day_of_Week'],
            hour=prediction_data['Hour'],
            weather_condition=prediction_data['Weather_Condition'],
            road_type=prediction_data['Road_Type'],
            junction_type=prediction_data['Junction_Type'],
            traffic_volume=prediction_data['Traffic_Volume'],
            speed_limit=prediction_data['Speed_Limit'],
            predicted_severity=prediction_data['Prediction'],
            confidence=prediction_data['Confidence']
        )
        session.add(record)
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def get_prediction_history():
    """
    Get all prediction records from the database
    
    Returns:
        pandas.DataFrame: DataFrame with prediction history
    """
    session = Session()
    try:
        records = session.query(PredictionRecord).all()
        
        # Convert to DataFrame
        data = []
        for record in records:
            data.append({
                'Day_of_Week': record.day_of_week,
                'Hour': record.hour,
                'Weather_Condition': record.weather_condition,
                'Road_Type': record.road_type,
                'Junction_Type': record.junction_type,
                'Traffic_Volume': record.traffic_volume,
                'Speed_Limit': record.speed_limit,
                'Prediction': record.predicted_severity,
                'Confidence': record.confidence,
                'Timestamp': record.timestamp
            })
        
        return pd.DataFrame(data)
    finally:
        session.close()

def save_sample_data(df):
    """
    Save sample traffic incident data to the database
    
    Args:
        df: DataFrame with sample data
    """
    session = Session()
    try:
        # Clear existing data
        session.query(TrafficIncident).delete()
        
        # Add new records
        for _, row in df.iterrows():
            record = TrafficIncident(
                timestamp=row['Timestamp'],
                day_of_week=row['Day_of_Week'],
                hour=row['Hour'],
                weather_condition=row['Weather_Condition'],
                road_type=row['Road_Type'],
                junction_type=row['Junction_Type'],
                traffic_volume=row['Traffic_Volume'],
                speed_limit=row['Speed_Limit'],
                severity=row['Severity'],
                severity_numeric=row['Severity_Numeric'],
                clearance_time=row['Clearance_Time']
            )
            session.add(record)
        
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def get_traffic_incidents():
    """
    Get all traffic incident records from the database
    
    Returns:
        pandas.DataFrame: DataFrame with traffic incidents
    """
    session = Session()
    try:
        records = session.query(TrafficIncident).all()
        
        # Convert to DataFrame
        data = []
        for record in records:
            data.append({
                'Timestamp': record.timestamp,
                'Day_of_Week': record.day_of_week,
                'Hour': record.hour,
                'Weather_Condition': record.weather_condition,
                'Road_Type': record.road_type,
                'Junction_Type': record.junction_type,
                'Traffic_Volume': record.traffic_volume,
                'Speed_Limit': record.speed_limit,
                'Severity': record.severity,
                'Severity_Numeric': record.severity_numeric,
                'Clearance_Time': record.clearance_time
            })
        
        return pd.DataFrame(data)
    finally:
        session.close()

# Initialize database schema when the module is imported
init_db()