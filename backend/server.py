from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime
import pandas as pd
import numpy as np
import io
import json
import base64
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Define Models
class StatusCheck(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class StatusCheckCreate(BaseModel):
    client_name: str

class EDAResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    analysis: Dict[str, Any]
    visualizations: Dict[str, str]  # base64 encoded images
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class EDAEngine:
    """Custom EDA Engine for analyzing CSV datasets"""
    
    def __init__(self, df: pd.DataFrame, filename: str):
        self.df = df
        self.filename = filename
        self.analysis = {}
        self.visualizations = {}
    
    def basic_info(self):
        """Get basic information about the dataset"""
        return {
            'shape': [int(x) for x in self.df.shape],
            'columns': list(self.df.columns),
            'dtypes': {k: str(v) for k, v in self.df.dtypes.to_dict().items()},
            'memory_usage': int(self.df.memory_usage(deep=True).sum()),
            'null_counts': {k: int(v) for k, v in self.df.isnull().sum().to_dict().items()},
            'null_percentages': {k: float(v) for k, v in (self.df.isnull().sum() / len(self.df) * 100).round(2).to_dict().items()}
        }
    
    def summary_statistics(self):
        """Generate summary statistics for numerical columns"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return {}
        
        stats_df = self.df[numeric_cols].describe()
        # Convert all values to native Python types
        result = {}
        for col in stats_df.columns:
            result[col] = {}
            for stat in stats_df.index:
                value = stats_df.loc[stat, col]
                if pd.isna(value):
                    result[col][stat] = None
                else:
                    result[col][stat] = float(value)
        return result
    
    def categorical_analysis(self):
        """Analyze categorical columns"""
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        cat_analysis = {}
        
        for col in categorical_cols:
            if col in self.df.columns:
                value_counts = self.df[col].value_counts()
                cat_analysis[col] = {
                    'unique_count': int(self.df[col].nunique()),
                    'top_values': {k: int(v) for k, v in value_counts.head(10).to_dict().items()},
                    'missing_count': int(self.df[col].isnull().sum())
                }
        
        return cat_analysis
    
    def correlation_analysis(self):
        """Calculate correlation matrix for numerical columns"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return {}
        
        corr_matrix = self.df[numeric_cols].corr()
        # Convert to native Python types
        result = {}
        for col in corr_matrix.columns:
            result[col] = {}
            for idx in corr_matrix.index:
                value = corr_matrix.loc[idx, col]
                if pd.isna(value):
                    result[col][idx] = None
                else:
                    result[col][idx] = float(value)
        return result
    
    def outlier_detection(self):
        """Detect outliers using IQR method"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        outliers = {}
        
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_count = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
            outliers[col] = {
                'count': int(outlier_count),
                'percentage': float(outlier_count / len(self.df) * 100),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound)
            }
        
        return outliers
    
    def create_visualizations(self):
        """Create various visualizations"""
        plt.style.use('default')
        
        # 1. Missing Values Heatmap
        if self.df.isnull().sum().sum() > 0:
            plt.figure(figsize=(12, 8))
            sns.heatmap(self.df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
            plt.title('Missing Values Heatmap')
            plt.tight_layout()
            self.visualizations['missing_values_heatmap'] = self._fig_to_base64()
        
        # 2. Correlation Heatmap
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            plt.figure(figsize=(10, 8))
            corr_matrix = self.df[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, fmt='.2f')
            plt.title('Correlation Heatmap')
            plt.tight_layout()
            self.visualizations['correlation_heatmap'] = self._fig_to_base64()
        
        # 3. Distribution plots for numerical columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Select up to 6 numerical columns for histograms
            cols_to_plot = list(numeric_cols)[:6]
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            for i, col in enumerate(cols_to_plot):
                if i < len(axes):
                    self.df[col].hist(bins=30, ax=axes[i], alpha=0.7)
                    axes[i].set_title(f'Distribution of {col}')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Frequency')
            
            # Hide unused subplots
            for i in range(len(cols_to_plot), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            self.visualizations['histograms'] = self._fig_to_base64()
        
        # 4. Box plots for numerical columns
        if len(numeric_cols) > 0:
            cols_to_plot = list(numeric_cols)[:6]
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            for i, col in enumerate(cols_to_plot):
                if i < len(axes):
                    self.df.boxplot(column=col, ax=axes[i])
                    axes[i].set_title(f'Box Plot of {col}')
            
            # Hide unused subplots
            for i in range(len(cols_to_plot), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            self.visualizations['boxplots'] = self._fig_to_base64()
        
        # 5. Data Types Distribution
        dtypes_count = self.df.dtypes.value_counts()
        plt.figure(figsize=(8, 6))
        plt.pie(dtypes_count.values, labels=dtypes_count.index, autopct='%1.1f%%', startangle=90)
        plt.title('Data Types Distribution')
        self.visualizations['dtypes_distribution'] = self._fig_to_base64()
    
    def _fig_to_base64(self):
        """Convert matplotlib figure to base64 string"""
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        return f"data:image/png;base64,{image_base64}"
    
    def run_analysis(self):
        """Run complete EDA analysis"""
        try:
            # Basic Information
            self.analysis['basic_info'] = self.basic_info()
            
            # Summary Statistics
            self.analysis['summary_statistics'] = self.summary_statistics()
            
            # Categorical Analysis
            self.analysis['categorical_analysis'] = self.categorical_analysis()
            
            # Correlation Analysis
            self.analysis['correlation_analysis'] = self.correlation_analysis()
            
            # Outlier Detection
            self.analysis['outlier_detection'] = self.outlier_detection()
            
            # Create Visualizations
            self.create_visualizations()
            
            return {
                'analysis': self.analysis,
                'visualizations': self.visualizations
            }
        
        except Exception as e:
            raise Exception(f"Error during EDA analysis: {str(e)}")

# Add your routes to the router instead of directly to app
@api_router.get("/")
async def root():
    return {"message": "Auto EDA API is running!"}

@api_router.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    """Upload and analyze CSV file"""
    try:
        # Validate file type
        if not file.filename.lower().endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        # Read file content
        content = await file.read()
        
        # Create DataFrame
        try:
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading CSV file: {str(e)}")
        
        # Validate DataFrame
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file is empty")
        
        if len(df.columns) == 0:
            raise HTTPException(status_code=400, detail="CSV file has no columns")
        
        # Run EDA Analysis
        eda_engine = EDAEngine(df, file.filename)
        results = eda_engine.run_analysis()
        
        # Create EDA result object
        eda_result = EDAResult(
            filename=file.filename,
            analysis=results['analysis'],
            visualizations=results['visualizations']
        )
        
        # Store in MongoDB
        await db.eda_results.insert_one(eda_result.dict())
        
        return {
            "id": eda_result.id,
            "filename": eda_result.filename,
            "analysis": eda_result.analysis,
            "visualizations": eda_result.visualizations,
            "message": "EDA analysis completed successfully"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in upload_csv: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@api_router.get("/eda-results")
async def get_eda_results():
    """Get all EDA results"""
    try:
        results = await db.eda_results.find({}, {"_id": 0}).sort("timestamp", -1).to_list(100)
        return {"results": results}
    except Exception as e:
        logging.error(f"Error in get_eda_results: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving EDA results")

@api_router.get("/eda-results/{result_id}")
async def get_eda_result(result_id: str):
    """Get specific EDA result by ID"""
    try:
        result = await db.eda_results.find_one({"id": result_id}, {"_id": 0})
        if not result:
            raise HTTPException(status_code=404, detail="EDA result not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in get_eda_result: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving EDA result")

@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.dict()
    status_obj = StatusCheck(**status_dict)
    _ = await db.status_checks.insert_one(status_obj.dict())
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    status_checks = await db.status_checks.find().to_list(1000)
    return [StatusCheck(**status_check) for status_check in status_checks]

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()