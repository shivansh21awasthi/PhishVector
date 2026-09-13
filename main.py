from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import joblib
import re
from lime.lime_text import LimeTextExplainer
from datetime import datetime

# --- DATABASE IMPORTS ---
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base, Session

# --- DATABASE SETUP ---
SQLALCHEMY_DATABASE_URL = "sqlite:///./phishvector.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Define the Database Table
class ScanLog(Base):
    __tablename__ = "scan_logs"
    id = Column(Integer, primary_key=True, index=True)
    email_text = Column(String, index=True)
    is_phishing = Column(Boolean)
    confidence_score = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)

# Create the table in the local directory
Base.metadata.create_all(bind=engine)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- APP & ML SETUP ---
app = FastAPI(title="PhishVector API", description="Phishing detection with Explainable AI & Database Logging")

# Load trained artifacts
model = joblib.load("models/phishing_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

class_names = ["Safe", "Phishing"]
explainer = LimeTextExplainer(class_names=class_names)

class EmailRequest(BaseModel):
    text: str

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return " ".join(text.split())

def predict_probabilities(texts):
    cleaned = [clean_text(t) for t in texts]
    vectors = vectorizer.transform(cleaned)
    return model.predict_proba(vectors)

# --- API ENDPOINTS ---

@app.post("/predict")
def predict_threat(request: EmailRequest, db: Session = Depends(get_db)):
    raw_text = request.text.strip()
    if len(raw_text) < 5:
        raise HTTPException(status_code=400, detail="Input text too short for analysis.")

    # Model inference
    cleaned = clean_text(raw_text)
    vector = vectorizer.transform([cleaned])
    probabilities = model.predict_proba(vector)[0]
    
    is_phishing = bool(model.predict(vector)[0] == 1)
    confidence = float(probabilities[1] if is_phishing else probabilities[0])

    # Generate LIME explanation
    exp = explainer.explain_instance(raw_text, predict_probabilities, num_features=6, labels=[1])
    feature_weights = [{"word": word, "weight": round(weight, 4)} for word, weight in exp.as_list(label=1)]

    # --- SAVE TO DATABASE ---
    new_log = ScanLog(
        email_text=raw_text,
        is_phishing=is_phishing,
        confidence_score=round(confidence * 100, 2)
    )
    db.add(new_log)
    db.commit()
    db.refresh(new_log)

    return {
        "log_id": new_log.id,
        "prediction": "Phishing/Spam Detected" if is_phishing else "Safe",
        "is_phishing": is_phishing,
        "confidence_score": round(confidence * 100, 2),
        "explanation": feature_weights
    }

@app.get("/logs")
def get_scan_history(limit: int = 10, db: Session = Depends(get_db)):
    """Retrieve the most recent threat scans from the database."""
    logs = db.query(ScanLog).order_by(ScanLog.timestamp.desc()).limit(limit).all()
    return logs