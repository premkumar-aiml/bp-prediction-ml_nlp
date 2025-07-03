def get_bp_category(diastolic_bp):
    if diastolic_bp <= 60:
        return {
            "status": "your blood pressure is low",
            "precaution": "Increase fluid and salt intake; consult a physician if dizziness occurs."
        }
    elif diastolic_bp < 80:
        return {
            "status": "your blood pressure is normal",
            "precaution": "Maintain a healthy lifestyle with regular exercise and a balanced diet."
        }
    elif 80 <= diastolic_bp <= 89:
        return {
            "status": "There’s a possibility of having Hypertension Stage I.",
            "precaution": "Reduce sodium intake, monitor BP regularly, and consult your doctor."
        }
    elif 90 <= diastolic_bp <= 119:
        return {
            "status": "There’s a possibility of having Hypertension Stage II",
            "precaution": "Follow prescribed medication and lifestyle changes; regular check-ups advised."
        }
    elif diastolic_bp >= 120:
        return {
            "status": "There’s a possibility of having Hypertensive Crisis",
            "precaution": "Seek emergency medical attention immediately."
        }
    else:
        return {
            "status": "Your blood pressure is Normal",
            "precaution": "Maintain a healthy lifestyle with regular exercise and a balanced diet"
        }
