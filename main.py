import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache
import warnings
import os
import json
import time
import requests
from datetime import datetime

warnings.filterwarnings('ignore')
os.environ['PYARROW_IGNORE_TIMEZONE'] = '1'

# ===================== AUTHENTICATION & LOGIN =====================

def init_session_state():
    """Initialize session state variables"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user_email' not in st.session_state:
        st.session_state.user_email = None
    if 'user_name' not in st.session_state:
        st.session_state.user_name = None

def save_user_credentials(email, name):
    """Save user credentials to local storage"""
    try:
        creds_dir = 'user_credentials'
        if not os.path.exists(creds_dir):
            os.makedirs(creds_dir)
        
        creds_file = os.path.join(creds_dir, 'saved_users.json')
        
        if os.path.exists(creds_file):
            with open(creds_file, 'r') as f:
                users = json.load(f)
        else:
            users = {}
        
        users[email] = {
            'name': name,
            'last_login': datetime.now().isoformat(),
            'login_count': users.get(email, {}).get('login_count', 0) + 1
        }
        
        with open(creds_file, 'w') as f:
            json.dump(users, f, indent=2)
        
        return True
    except Exception as e:
        st.error(f"Error saving credentials: {str(e)}")
        return False

def load_saved_users():
    """Load saved user credentials"""
    try:
        creds_file = os.path.join('user_credentials', 'saved_users.json')
        if os.path.exists(creds_file):
            with open(creds_file, 'r') as f:
                return json.load(f)
        return {}
    except:
        return {}

def send_teams_notification(user_email, user_name):
    """Send notification to Microsoft Teams"""
    try:
        from teams_config import TEAMS_WEBHOOK_URL
        
        message = {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "summary": "New User Login",
            "themeColor": "D70000",
            "sections": [{
                "activityTitle": "SKU Classification System - New Login",
                "activitySubtitle": f"User: {user_name} ({user_email})",
                "facts": [
                    {"name": "Email", "value": user_email},
                    {"name": "Name", "value": user_name},
                    {"name": "Login Time", "value": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                ]
            }]
        }
        
        response = requests.post(TEAMS_WEBHOOK_URL, json=message)
        return response.status_code == 200
        
    except Exception as e:
        print(f"Teams notification error: {str(e)}")
        return True

def validate_email(email):
    """Validate if email is from allowed domains"""
    allowed_domains = ['@thermofisher.com']
    return any(domain in email.lower() for domain in allowed_domains)

def login_page():
    """Display login page"""
    st.title("🔐 SKU Classification System - Login")
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.subheader("Please Login to Continue")
        
        # Check for saved users
        saved_users = load_saved_users()
        
        if saved_users:
            st.info("💡 Select from saved users or login with email")
            
            # Dropdown for saved users
            saved_emails = list(saved_users.keys())
            selected_email = st.selectbox("Saved Users", ["Select a user..."] + saved_emails)
            
            if selected_email != "Select a user..." and st.button("Login as Saved User", use_container_width=True):
                user_data = saved_users[selected_email]
                st.session_state.authenticated = True
                st.session_state.user_email = selected_email
                st.session_state.user_name = user_data['name']
                
                # Update login count
                save_user_credentials(selected_email, user_data['name'])
                send_teams_notification(selected_email, user_data['name'])
                
                st.success(f"Welcome back, {user_data['name']}!")
                st.rerun()
            
            st.markdown("---")
            st.markdown("**Or login with email address:**")
        
        # New login form
        with st.form("login_form"):
            email = st.text_input(
                "Email Address",
                placeholder="Please enter your ThermoFisher email",
                help="Use your ThermoFisher email address"
            )
            
            name = st.text_input(
                "Full Name",
                placeholder="Full Name",
                help="Enter your full name"
            )
            
            login_submitted = st.form_submit_button("🚀 Login", use_container_width=True)
        
        if login_submitted:
            if not email or not name:
                st.error("Please fill in both email and name fields")
            elif not validate_email(email):
                st.error("Please use a valid ThermoFisher (@thermofisher.com) email address")
            else:
                # Save credentials and authenticate
                if save_user_credentials(email, name):
                    st.session_state.authenticated = True
                    st.session_state.user_email = email
                    st.session_state.user_name = name
                    
                    # Send Teams notification
                    send_teams_notification(email, name)
                    
                    st.success(f"Welcome, {name}! Login successful.")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Error during login. Please try again.")
        
        # Instructions
        st.markdown("---")
        st.info("""
        **Allowed Email Domains:**
        - @thermofisher.com
        
        Your credentials will be saved for future logins.
        """)

# ===================== FEEDBACK SYSTEM =====================

def save_feedback(sku_input, name_input, prediction, feedback_type, user_email, correct_product_line=None, correct_business_unit=None, user_comment=None):
    """Save user feedback with user information and send Teams notification"""
    try:
        feedback_dir = 'feedback_data'
        if not os.path.exists(feedback_dir):
            os.makedirs(feedback_dir)
        
        feedback_file = os.path.join(feedback_dir, 'bulk_user_feedback.json')
        feedback_csv_file = os.path.join(feedback_dir, 'bulk_user_feedback.csv')
        
        feedback_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'user_email': user_email,
            'sku_input': sku_input,
            'name_input': name_input,
            'predicted_product_line_code': prediction.get('product_line_code', ''),
            'predicted_cmr_line': prediction.get('cmr_product_line', ''),
            'predicted_product_line_name': prediction.get('product_line_name', ''),
            'predicted_sub_platform': prediction.get('sub_platform', ''),
            'predicted_sku_number': prediction.get('sku_number', ''),
            'predicted_sku_name': prediction.get('sku_name', ''),
            'confidence_score': prediction.get('combined_score', 0),
            'sku_score': prediction.get('sku_score', 0),
            'name_score': prediction.get('name_score', 0),
            'feedback': feedback_type,
            'prediction_type': 'exact' if prediction.get('combined_score', 0) == 100 else 'fuzzy',
            'correct_product_line_code': correct_product_line or '',
            'correct_business_unit': correct_business_unit or '',
            'user_comment': user_comment or ''
        }
        
        # Save to JSON
        if os.path.exists(feedback_file):
            with open(feedback_file, 'r') as f:
                feedback_list = json.load(f)
        else:
            feedback_list = []
        
        feedback_list.append(feedback_data)
        
        with open(feedback_file, 'w') as f:
            json.dump(feedback_list, f, indent=2)
        
        # Save to CSV
        feedback_df = pd.DataFrame([feedback_data])
        
        if os.path.exists(feedback_csv_file):
            existing_df = pd.read_csv(feedback_csv_file)
            combined_df = pd.concat([existing_df, feedback_df], ignore_index=True)
            combined_df.to_csv(feedback_csv_file, index=False)
        else:
            feedback_df.to_csv(feedback_csv_file, index=False)
        
        # Send Teams notification for feedback
        try:
            from teams_config import send_feedback_notification
            user_name = st.session_state.user_name
            send_feedback_notification(user_email, user_name, sku_input, name_input, prediction, feedback_type, correct_product_line, correct_business_unit, user_comment)
        except Exception as teams_error:
            print(f"Teams notification error: {str(teams_error)}")
        
        return True
    except Exception as e:
        st.error(f"Error saving feedback: {str(e)}")
        return False

def load_feedback_data():
    """Load feedback data for analysis"""
    feedback_file = os.path.join('feedback_data', 'bulk_user_feedback.csv')
    if os.path.exists(feedback_file):
        return pd.read_csv(feedback_file)
    return pd.DataFrame()

def get_unique_product_line_codes(df):
    """Get unique product line codes for dropdown"""
    try:
        # Convert to string and remove NaN values
        codes = df['product line code'].dropna().astype(str).unique().tolist()
        # Filter out empty strings and sort
        codes = [code for code in codes if code.strip()]
        return sorted(codes)
    except Exception as e:
        print(f"Error getting product line codes: {str(e)}")
        return []

def get_unique_business_units(df):
    """Get unique business units for dropdown"""
    try:
        # Convert to string and remove NaN values
        units = df['sub platform'].dropna().astype(str).unique().tolist()
        # Filter out empty strings and sort
        units = [unit for unit in units if unit.strip()]
        return sorted(units)
    except Exception as e:
        print(f"Error getting business units: {str(e)}")
        return []

def get_training_data_with_feedback():
    """Convert feedback to training data format"""
    feedback_df = load_feedback_data()
    
    if feedback_df.empty:
        return None
    
    training_data_list = []
    
    # Process LIKES - use predicted data (confirmed correct)
    positive_feedback = feedback_df[feedback_df['feedback'] == 'like'].copy()
    if not positive_feedback.empty:
        likes_data = pd.DataFrame({
            'sku number': positive_feedback['sku_input'],
            'sku name': positive_feedback['name_input'],
            'product line code': positive_feedback['predicted_product_line_code'],
            'cmr product line': positive_feedback['predicted_cmr_line'],
            'product line name': positive_feedback['predicted_product_line_name'],
            'sub platform': positive_feedback['predicted_sub_platform']
        })
        training_data_list.append(likes_data)
    
    # Process DISLIKES with corrections - use corrected data
    negative_feedback = feedback_df[feedback_df['feedback'] == 'dislike'].copy()
    if not negative_feedback.empty:
        # Filter dislikes that have corrections provided
        corrected_feedback = negative_feedback[
            (negative_feedback['correct_product_line_code'].notna() & 
             negative_feedback['correct_product_line_code'] != '') |
            (negative_feedback['correct_business_unit'].notna() & 
             negative_feedback['correct_business_unit'] != '')
        ].copy()
        
        if not corrected_feedback.empty:
            # Use corrected data for training
            corrections_data = pd.DataFrame({
                'sku number': corrected_feedback['sku_input'],
                'sku name': corrected_feedback['name_input'],
                'product line code': corrected_feedback['correct_product_line_code'].fillna(corrected_feedback['predicted_product_line_code']),
                'cmr product line': corrected_feedback['predicted_cmr_line'],  # Keep original CMR for now
                'product line name': corrected_feedback['predicted_product_line_name'],  # Will be updated based on code
                'sub platform': corrected_feedback['correct_business_unit'].fillna(corrected_feedback['predicted_sub_platform'])
            })
            training_data_list.append(corrections_data)
    
    if not training_data_list:
        return None
    
    # Combine all training data
    combined_training_data = pd.concat(training_data_list, ignore_index=True)
    return combined_training_data

def retrain_with_feedback():
    """Retrain model incorporating feedback data (likes + corrected dislikes)"""
    original_df = load_data()
    feedback_training_data = get_training_data_with_feedback()
    
    if feedback_training_data is not None and not feedback_training_data.empty:
        # Get feedback statistics
        feedback_df = load_feedback_data()
        likes_count = len(feedback_df[feedback_df['feedback'] == 'like'])
        corrected_dislikes = len(feedback_df[
            (feedback_df['feedback'] == 'dislike') & 
            ((feedback_df['correct_product_line_code'].notna() & feedback_df['correct_product_line_code'] != '') |
             (feedback_df['correct_business_unit'].notna() & feedback_df['correct_business_unit'] != ''))
        ])
        
        # Combine original data with feedback data
        combined_df = pd.concat([original_df, feedback_training_data], ignore_index=True)
        combined_df = combined_df.drop_duplicates()
        
        st.success(f"✅ Model Enhanced Successfully!")
        st.info(f"📊 Added {len(feedback_training_data)} training samples:")
        st.info(f"   • {likes_count} confirmed correct predictions (likes)")
        st.info(f"   • {corrected_dislikes} user corrections (corrected dislikes)")
        
        return combined_df
    else:
        st.info("No feedback data available for retraining")
        return original_df

# ===================== CORE DATA FUNCTIONS =====================

@st.cache_data(ttl=3600)
def load_data():
    """Load and preprocess training data"""
    try:
        df = pd.read_excel('Training_Set.xlsx')
        
        # Clean data - remove empty rows
        df = df[(df['sku number'].notna()) & (df['sku number'] != '') & 
                (df['sku name'].notna()) & (df['sku name'] != '')]
        
        # Filter valid product lines
        valid_cmr_product_lines = [
            'BEAService', 'BEAHardware', 'BEAOther', 'HardwareConsumables', 
            'SUTAutomation','2DBioProcessContainers', '3DBioProcessContainers', 
            'FillFinish', 'FlexibleOther','FluidTransferAssemblies', 
            'BioproductionContainments', 'BottleAssemblies',
            'ProductionCellCulture', 'RigidOther', 'SUDOther'
        ]
        df = df[df['cmr product line'].isin(valid_cmr_product_lines)]
        
        # Remove duplicates and clean SKU numbers
        df = df.drop_duplicates(subset=['sku number', 'sku name'])
        df['sku number'] = df['sku number'].str.replace(r'(INT_FINESS.*|BPD.*)', '', regex=True)
        
        # Apply volume-based CMR product line correction during data loading
        df['cmr product line'] = df.apply(lambda row: determine_correct_cmr_by_volume(
            row['sku name'], row['cmr product line']), axis=1)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_data
def load_business_rules_data():
    """Load business rule book data"""
    try:
        return pd.read_excel('Business_Rule.xlsx')
    except Exception as e:
        st.error(f"Error loading Business Rule Book: {str(e)}")
        return None

@st.cache_data
def load_reference_hierarchy():
    """Load reference hierarchy data for Product Line - LV 2 mapping"""
    try:
        return pd.read_excel('reference_file_hierechy.xlsx')
    except Exception as e:
        st.error(f"Error loading reference hierarchy: {str(e)}")
        return None

def get_product_line_lv2(product_line_code, df_hierarchy):
    """Get Product Line - LV 2 from product line code"""
    if df_hierarchy is None:
        return 'N/A'
    
    try:
        match = df_hierarchy[df_hierarchy['PL Codes'] == product_line_code]
        if not match.empty:
            return match.iloc[0]['Product Line - LV 2']
        return 'N/A'
    except Exception:
        return 'N/A'

@st.cache_resource
def create_similarity_index(df):
    """Create TF-IDF similarity index"""
    combined_text = (df['sku number'].astype(str) + " " + 
                    df['sku name'].astype(str)).str.lower()
    
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=3000,
        analyzer='char_wb',
        lowercase=True,
        min_df=1
    )
    tfidf_matrix = vectorizer.fit_transform(combined_text)
    return vectorizer, tfidf_matrix

# ===================== VOLUME EXTRACTION & 2D/3D MAPPING =====================

def extract_volume_enhanced(sku_name):
    """Enhanced volume extraction with comprehensive pattern matching"""
    if not sku_name:
        return None
    
    sku_name = str(sku_name).upper().strip()
    
    # Liter patterns
    liter_patterns = [
        r'(\d+(?:\.\d+)?)\s*L(?:\s|$|[^\w])',
        r'(\d+(?:\.\d+)?)\s*LITER',
        r'(\d+(?:\.\d+)?)\s*LITRE'
    ]
    
    for pattern in liter_patterns:
        match = re.search(pattern, sku_name)
        if match:
            return float(match.group(1))
    
    # Milliliter patterns
    ml_patterns = [
        r'(\d+(?:\.\d+)?)\s*M?ML(?:\s|$|[^\w])',
        r'(\d+(?:\.\d+)?)\s*MILLILITER',
        r'(\d+(?:\.\d+)?)\s*MILLILITRE'
    ]
    
    for pattern in ml_patterns:
        match = re.search(pattern, sku_name)
        if match:
            return float(match.group(1)) / 1000  # Convert to liters
    
    return None

def determine_correct_cmr_by_volume(sku_name, original_cmr_line):
    """Determine correct CMR product line based on 50L volume rule"""
    if original_cmr_line not in ['2DBioProcessContainers', '3DBioProcessContainers']:
        return original_cmr_line
    
    volume_l = extract_volume_enhanced(sku_name)
    if volume_l is None:
        return original_cmr_line
    
    return '2DBioProcessContainers' if volume_l <= 50 else '3DBioProcessContainers'

def create_2d_to_3d_mapping():
    """
    Create mapping from 2D product line codes to appropriate 3D codes.
    Based on analysis of the data structure.
    """
    return {
        '2JE': '2MH',  # GENERAL 2D -> PRODUCTAINER BPC
        '2JC': '2MH',  # LABTAINER -> PRODUCTAINER BPC
        '2PQ': '2PS',  # 2DBioProcessContainers Tieout -> 3DBioProcessContainers Tieout
        '2MD': '2MN',  # Map to 3D Manifold
        '2JD': '2MH',  # Map to PRODUCTAINER BPC
        '0CF': '0D8',  # 2D SINGLE -> SINGLE
        '2MB': '2MH',  # Map to PRODUCTAINER BPC
        '2MF': '2MJ',  # 2D TANK LINER -> 3D TANK LINERS
        '0D0': '2MN',  # MANIFOLD -> 3D MANIFOLD
        'Z3U': '2MH',  # Map to PRODUCTAINER BPC
        'Z6R': '0D8',  # 2D SINGLE -> SINGLE
        '0CZ': '0D8',  # 2D SINGLE -> SINGLE
        'Z3R': '0D8',  # 2D SINGLE -> SINGLE
        'Z2K': '2MN',  # MANIFOLD -> 3D MANIFOLD
        'Z37': '0D8'   # 2D SINGLE -> SINGLE
    }

def get_2d_to_3d_mapping():
    # Duplicate of create_2d_to_3d_mapping, remove to avoid confusion
    pass

def get_3d_to_2d_mapping():
    """3D to 2D product line code mapping"""
    return {
        '2MH': '2JE', '2MJ': '2MF', '2MO': '2JE', '2PS': '2JE', '2MN': '2MD',
        'Z2H': '2JE', '2ML': '2JE', 'Z39': '2JE', 'Z6M': '2JE', '2MM': '2JE',
        '0D8': '2MD', '0EG': '2JE', '3D6': '2JE', '3WO': '2JE', 'Z3Q': '2JE',
        '262': '2JE', '2MG': '2JE'
    }

def get_product_line_name(product_line_code, is_2d=True):
    """Get appropriate product line name"""
    if is_2d:
        mapping = {
            '2JE': 'FLEXIBLE CONSUMABLES 2D', '2JC': 'LABTAINER', '2JD': 'GENERAL 2D',
            '2PQ': '2DBioProcessContainers Tieout', '2MD': '2D MANIFOLD', '0CF': '2D SINGLE',
            '2MB': '2D HARVESTAINER', '2MF': '2D TANK LINER', '0D0': '2D MANIFOLD',
            'Z3U': 'MANIFOLD', 'Z6R': '2D SINGLE', '0CZ': '2D MANIFOLD', 'Z3R': '2D SINGLE',
            'Z2K': 'MANIFOLD', 'Z37': 'FLEXIBLE CONSUMABLES 2D'
        }
        return mapping.get(product_line_code, 'FLEXIBLE CONSUMABLES 2D')
    else:
        mapping = {
            '2MH': 'PRODUCTAINER BPC', '2MJ': '3D TANK LINERS', '2MO': '3D PRODUCTAINER',
            '2PS': '3D PRODUCTAINER', '2MN': '3D MANIFOLD', 'Z2H': '3D PRODUCTAINER',
            '2ML': '3D PRODUCTAINER', 'Z39': '3D PRODUCTAINER', 'Z6M': '3D PRODUCTAINER',
            '2MM': 'OTHER OUTER SUPPORT CONTAINERS', '0D8': '3D MANIFOLD', '0EG': '3D PRODUCTAINER',
            '3D6': '3D PRODUCTAINER', '3WO': '3D PRODUCTAINER', 'Z3Q': '3D PRODUCTAINER',
            '262': '3D PRODUCTAINER', '2MG': '3D PRODUCTAINER'
        }
        return mapping.get(product_line_code, 'PRODUCTAINER BPC')

def adjust_product_line_for_volume(original_cmr, product_line_code, product_line_name, sku_name):
    """Adjust product line code and name based on volume-determined CMR classification"""
    sku_name_upper = str(sku_name).upper() if sku_name else ""
    if "SPIGOT NEEDLE" in sku_name_upper:
        return "2L0", "BE20 TANK FITTINGS", "BioproductionContainments"
    if "PILLOW BAG" in sku_name_upper:
        return "2JE", "GENERAL 2D", "2DBioProcessContainers"
    if "BETA BAG" in sku_name_upper or "NEEDLE" in sku_name_upper:
        return "2NK", "FF FILLING ASSEMBLIES", "FillFinish"
    if original_cmr not in ['2DBioProcessContainers', '3DBioProcessContainers']:
        return product_line_code, product_line_name, original_cmr

    correct_cmr = determine_correct_cmr_by_volume(sku_name, original_cmr)

    # Always map 2D code to 3D code if CMR is 3DBioProcessContainers
    if correct_cmr == '3DBioProcessContainers':
        mapped_code = create_2d_to_3d_mapping().get(product_line_code, product_line_code)
        mapped_name = get_product_line_name(mapped_code, is_2d=False)
        return mapped_code, mapped_name, correct_cmr
    # Always map 3D code to 2D code if CMR is 2DBioProcessContainers
    elif correct_cmr == '2DBioProcessContainers':
        mapped_code = get_3d_to_2d_mapping().get(product_line_code, product_line_code)
        mapped_name = get_product_line_name(mapped_code, is_2d=True)
        return mapped_code, mapped_name, correct_cmr
    # Fallback (should not hit)
    return product_line_code, product_line_name, correct_cmr

# ===================== SIMILARITY & PREDICTION FUNCTIONS =====================

@lru_cache(maxsize=500)
def calculate_simple_similarity(s1, s2):
    """Cached similarity calculation"""
    if not s1 or not s2:
        return 0.0
    
    s1, s2 = s1.lower(), s2.lower()
    if s1 in s2 or s2 in s1:
        return 85.0
    
    set1, set2 = set(s1), set(s2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return (intersection / union) * 100 if union > 0 else 0.0

def get_exact_predictions(df, sku_partial, name_partial):
    """Find exact substring matches"""
    if not sku_partial.strip() and not name_partial.strip():
        return pd.DataFrame()
    
    df_copy = df.copy()
    
    if sku_partial.strip():
        df_copy = df_copy[df_copy['sku number'].astype(str).str.upper().str.contains(
            sku_partial.upper(), na=False, regex=False)]
    
    if name_partial.strip():
        df_copy = df_copy[df_copy['sku name'].astype(str).str.lower().str.contains(
            name_partial.lower(), na=False, regex=False)]
    
    return df_copy.drop_duplicates(
        subset=['sku number', 'sku name', 'product line code', 'cmr product line']
    ).head(10)

def get_fuzzy_predictions(df, sku_partial, name_partial, vectorizer, tfidf_matrix, top_k=5):
    """Get fuzzy predictions with volume-based 2D/3D mapping"""
    if not sku_partial.strip() and not name_partial.strip():
        return []
    
    query_text = f"{sku_partial} {name_partial}".lower()
    query_vector = vectorizer.transform([query_text])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    top_indices = np.argsort(similarities)[-top_k*3:][::-1]
    results = []
    seen_combinations = set()
    
    # Load hierarchy data for Product Line - LV 2
    df_hierarchy = st.session_state.get('df_hierarchy')
    
    for idx in top_indices:
        if len(results) >= top_k or similarities[idx] < 0.1:
            continue
            
        row = df.iloc[idx]
        combination_id = f"{row['product line code']}|{row['cmr product line']}"
        
        if combination_id not in seen_combinations:
            seen_combinations.add(combination_id)
            
            # Apply volume-based mapping using input name
            adj_code, adj_name, correct_cmr = adjust_product_line_for_volume(
                row['cmr product line'], row['product line code'], 
                row['product line name'], name_partial
            )
            
            # Get Product Line - LV 2
            product_line_lv2 = get_product_line_lv2(adj_code, df_hierarchy)
            
            results.append({
                'sku_number': row['sku number'],
                'sku_name': row['sku name'],
                'product_line_code': adj_code,
                'cmr_product_line': correct_cmr,
                'product_line_name': adj_name,
                'product_line_lv2': product_line_lv2,
                'sub_platform': row['sub platform'],
                'sku_score': round(calculate_simple_similarity(sku_partial, str(row['sku number'])), 2),
                'name_score': round(calculate_simple_similarity(name_partial, str(row['sku name'])), 2),
                'combined_score': round(similarities[idx] * 100, 2)
            })
    
    return results

def ultra_fast_bulk_predictions(input_df, df, vectorizer, tfidf_matrix):
    """Ultra-fast bulk prediction with optimized 2D/3D mapping"""
    # Prepare input data
    input_clean = input_df.copy()
    input_clean['sku number'] = input_clean['sku number'].fillna('').astype(str)
    input_clean['sku name'] = input_clean['sku name'].fillna('').astype(str)
    query_texts = (input_clean['sku number'] + ' ' + input_clean['sku name']).str.lower()
    
    # Vectorized similarity computation
    with st.spinner("Computing similarities..."):
        query_vectors = vectorizer.transform(query_texts)
        similarities = cosine_similarity(query_vectors, tfidf_matrix)
    
    # Find top 3 matches efficiently
    with st.spinner("Finding top matches..."):
        top_k = 3
        top_indices = np.argpartition(-similarities, range(top_k), axis=1)[:, :top_k]
        sorted_indices = np.argsort(-similarities[np.arange(similarities.shape[0])[:, None], top_indices], axis=1)
        final_indices = top_indices[np.arange(similarities.shape[0])[:, None], sorted_indices]
        top_scores = similarities[np.arange(similarities.shape[0])[:, None], final_indices]
    
    # Build results with volume-based mapping
    with st.spinner("Building results..."):
        results = []
        for i, (_, row) in enumerate(input_df.iterrows()):
            result_row = row.copy()
            
            for pred_num in range(3):
                if pred_num < final_indices.shape[1] and top_scores[i, pred_num] >= 0.1:
                    match_row = df.iloc[final_indices[i, pred_num]]
                    
                    # Apply volume-based mapping
                    adj_code, adj_name, correct_cmr = adjust_product_line_for_volume(
                        match_row['cmr product line'], match_row['product line code'],
                        match_row['product line name'], row.get('sku name', '')
                    )
                    
                    # Get Product Line - LV 2
                    df_hierarchy = st.session_state.get('df_hierarchy')
                    product_line_lv2 = get_product_line_lv2(adj_code, df_hierarchy)
                    
                    prefix = f'Prediction {pred_num+1}: '
                    result_row[f'{prefix}SKU Number'] = match_row['sku number']
                    result_row[f'{prefix}SKU Name'] = match_row['sku name']
                    result_row[f'{prefix}CMR Product Line'] = correct_cmr
                    result_row[f'{prefix}Product Line Name'] = adj_name
                    result_row[f'{prefix}Product Line Code'] = adj_code
                    result_row[f'{prefix}Product Line - LV 2'] = product_line_lv2
                    result_row[f'{prefix}Business Unit'] = match_row['sub platform']
                    result_row[f'{prefix}Confidence Score'] = round(top_scores[i, pred_num] * 100, 2)
                else:
                    prefix = f'Prediction {pred_num+1}: '
                    for suffix in ['SKU Number', 'SKU Name', 'CMR Product Line', 
                                 'Product Line Name', 'Product Line Code', 'Business Unit']:
                        result_row[f'{prefix}{suffix}'] = 'No Match Found'
                    result_row[f'{prefix}Confidence Score'] = 0.0
            
            results.append(result_row)
    
    return pd.DataFrame(results)

# ===================== BUSINESS RULES =====================

def get_business_rule(product_line_code, df_rules):
    """Get business rule for product line code"""
    if df_rules is None:
        return None, None
    
    try:
        rule_row = df_rules[df_rules['product line code'] == product_line_code]
        if not rule_row.empty:
            rule = rule_row.iloc[0]
            return (rule.get('Top Trigrams in SKU Name', 'N/A'),
                   rule.get('Top Prefixes in SKU No.', 'N/A'))
        return None, None
    except Exception as e:
        st.error(f"Error retrieving business rule: {str(e)}")
        return None, None

# ===================== DISPLAY FUNCTIONS =====================

def safe_dataframe_display(df, key=None):
    """Safe dataframe display with fallback"""
    try:
        st.dataframe(df, key=key)
    except Exception:
        st.warning("Display issue detected. Showing alternative preview.")
        st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        if len(df) > 0:
            st.json(df.head(3).to_dict('records'))

def display_exact_matches(exact_matches, sku_input, name_input, df):
    """Display exact match results with feedback buttons"""
    st.subheader("✅ Exact Matches Found")
    st.success(f"{len(exact_matches)} Strong Prediction(s) Found")
    
    for idx, row in exact_matches.iterrows():
        # Adjust code/name for 2D/3D CMR
        adj_code, adj_name, adj_cmr = adjust_product_line_for_volume(
            row['cmr product line'], row['product line code'], row['product line name'], row['sku name']
        )
        
        # Get Product Line - LV 2
        df_hierarchy = st.session_state.get('df_hierarchy')
        product_line_lv2 = get_product_line_lv2(adj_code, df_hierarchy)
        
        with st.expander(f"Match {idx + 1}: {adj_code} - {adj_cmr}", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**SKU Number:**", row['sku number'])
                st.write("**Product Line Code:**", adj_code)
                st.write("**Product Line Name:**", adj_name)
                st.write("**Product Line - LV 2:**", product_line_lv2)
            with col2:
                st.write("**SKU Name:**", row['sku name'])
                st.write("**CMR Product Line:**", adj_cmr)
                st.write("**Business Unit:**", row['sub platform'])
            
            # Feedback buttons
            st.markdown("**Is this prediction correct?**")
            feedback_col1, feedback_col2 = st.columns(2)
            
            prediction_data = {
                'product_line_code': adj_code,
                'cmr_product_line': adj_cmr,
                'product_line_name': adj_name,
                'sub_platform': row['sub platform'],
                'sku_number': row['sku number'],
                'sku_name': row['sku name'],
                'combined_score': 100
            }
            
            with feedback_col1:
                if st.button("👍 Like", key=f"like_exact_{idx}", use_container_width=True):
                    if save_feedback(sku_input, name_input, prediction_data, "like", st.session_state.user_email):
                        st.success("✅ Feedback submitted successfully! Thank you for your feedback.")
                        time.sleep(10)
            
            with feedback_col2:
                if st.button("👎 Dislike", key=f"dislike_exact_{idx}", use_container_width=True):
                    st.session_state[f'show_dislike_form_exact_{idx}'] = True
            
            # Enhanced dislike form
            if st.session_state.get(f'show_dislike_form_exact_{idx}', False):
                st.markdown("---")
                st.markdown("**📝 Please help us improve by providing the correct information:**")
                
                with st.form(f"dislike_form_exact_{idx}"):
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        # Get unique options from training data
                        product_line_options = get_unique_product_line_codes(df)
                        correct_product_line = st.selectbox(
                            "Correct Product Line Code:",
                            options=["Select..."] + product_line_options,
                            key=f"correct_pl_exact_{idx}"
                        )
                    
                    with col_b:
                        business_unit_options = get_unique_business_units(df)
                        correct_business_unit = st.selectbox(
                            "Correct Business Unit:",
                            options=["Select..."] + business_unit_options,
                            key=f"correct_bu_exact_{idx}"
                        )
                    
                    user_comment = st.text_area(
                        "Additional Comments (Optional):",
                        placeholder="Please provide any additional feedback or suggestions...",
                        key=f"comment_exact_{idx}"
                    )
                    
                    col_submit, col_cancel = st.columns(2)
                    with col_submit:
                        submit_dislike = st.form_submit_button("📝 Submit Feedback", use_container_width=True)
                    with col_cancel:
                        cancel_dislike = st.form_submit_button("❌ Cancel", use_container_width=True)
                
                if submit_dislike:
                    correct_pl = correct_product_line if correct_product_line != "Select..." else None
                    correct_bu = correct_business_unit if correct_business_unit != "Select..." else None
                    
                    if save_feedback(sku_input, name_input, prediction_data, "dislike", 
                                   st.session_state.user_email, correct_pl, correct_bu, user_comment):
                        st.success("✅ Feedback submitted successfully! Thank you for your detailed feedback.")
                        time.sleep(10)
                        st.session_state[f'show_dislike_form_exact_{idx}'] = False
                        st.rerun()
                
                if cancel_dislike:
                    st.session_state[f'show_dislike_form_exact_{idx}'] = False
                    st.rerun()

def display_fuzzy_matches(fuzzy_matches, sku_input, name_input, df_rules):
    """Display fuzzy match results with feedback buttons"""
    if not fuzzy_matches:
        st.warning("No fuzzy matches found above the threshold.")
        return
    
    st.markdown("### 📥 Input Parameters")
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Input SKU Number:** {sku_input if sku_input else 'N/A'}")
    with col2:
        st.info(f"**Input SKU Name:** {name_input if name_input else 'N/A'}")
    
    st.markdown("---")
    st.subheader("🔍 Top Fuzzy Predictions")
    
    for i, match in enumerate(fuzzy_matches, 1):
        confidence_color = "🟢" if match['combined_score'] >= 80 else "🟡" if match['combined_score'] >= 60 else "🔴"
        
        with st.expander(
            f"{confidence_color} Prediction {i}: {match['product_line_code']} - {match['cmr_product_line']} | "
            f"{match['product_line_name']} | {match['sub_platform']} "
            f"(Confidence: {match['combined_score']}%)",
            expanded=(i == 1)
        ):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**SKU Number:**", match['sku_number'])
                st.write("**Product Line Code:**", match['product_line_code'])
                st.write("**Product Line Name:**", match['product_line_name'])
                st.write("**Product Line - LV 2:**", match.get('product_line_lv2', 'N/A'))
            
            with col2:
                truncated_name = match['sku_name'][:50] + "..." if len(match['sku_name']) > 50 else match['sku_name']
                st.write("**SKU Name:**", truncated_name)
                st.write("**CMR Product Line:**", match['cmr_product_line'])
                st.write("**Business Unit:**", match['sub_platform'])
            
            with col3:
                st.metric("Confidence Score", f"{match['combined_score']}%")
            
            if len(match['sku_name']) > 50:
                st.write("**Full SKU Name:**", match['sku_name'])

            # Business Rule Section
            st.markdown("---")
            st.markdown("**📋 Business Rule Identified:**")
            
            sku_name_pattern, sku_prefix_pattern = get_business_rule(match['product_line_code'], df_rules)
            
            if sku_name_pattern and sku_prefix_pattern:
                st.info(f"**Common SKU Name Pattern:** {sku_name_pattern}")
                st.info(f"**Prefix Pattern Found:** {sku_prefix_pattern}")
            else:
                st.warning("No business rule found for this product line code")
            
            # Feedback buttons
            st.markdown("---")
            st.markdown("**Is this prediction correct?**")
            feedback_col1, feedback_col2 = st.columns(2)
            
            with feedback_col1:
                if st.button("👍 Like", key=f"like_fuzzy_{i}", use_container_width=True):
                    if save_feedback(sku_input, name_input, match, "like", st.session_state.user_email):
                        st.success("✅ Feedback submitted successfully! Thank you for your feedback.")
                        time.sleep(10)
            
            with feedback_col2:
                if st.button("👎 Dislike", key=f"dislike_fuzzy_{i}", use_container_width=True):
                    st.session_state[f'show_dislike_form_fuzzy_{i}'] = True
            
            # Enhanced dislike form
            if st.session_state.get(f'show_dislike_form_fuzzy_{i}', False):
                st.markdown("---")
                st.markdown("**📝 Please help us improve by providing the correct information:**")
                
                with st.form(f"dislike_form_fuzzy_{i}"):
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        # Get unique options from training data
                        product_line_options = get_unique_product_line_codes(st.session_state.df)
                        
                        correct_product_line = st.selectbox(
                            "Correct Product Line Code:",
                            options=["Select..."] + product_line_options,
                            key=f"correct_pl_fuzzy_{i}"
                        )
                    
                    with col_b:
                        business_unit_options = get_unique_business_units(st.session_state.df)
                        correct_business_unit = st.selectbox(
                            "Correct Business Unit:",
                            options=["Select..."] + business_unit_options,
                            key=f"correct_bu_fuzzy_{i}"
                        )
                    
                    user_comment = st.text_area(
                        "Additional Comments (Optional):",
                        placeholder="Please provide any additional feedback or suggestions...",
                        key=f"comment_fuzzy_{i}"
                    )
                    
                    col_submit, col_cancel = st.columns(2)
                    with col_submit:
                        submit_dislike = st.form_submit_button("📝 Submit Feedback", use_container_width=True)
                    with col_cancel:
                        cancel_dislike = st.form_submit_button("❌ Cancel", use_container_width=True)
                
                if submit_dislike:
                    correct_pl = correct_product_line if correct_product_line != "Select..." else None
                    correct_bu = correct_business_unit if correct_business_unit != "Select..." else None
                    
                    if save_feedback(sku_input, name_input, match, "dislike", 
                                   st.session_state.user_email, correct_pl, correct_bu, user_comment):
                        st.success("✅ Feedback submitted successfully! Thank you for your detailed feedback.")
                        time.sleep(10)
                        st.session_state[f'show_dislike_form_fuzzy_{i}'] = False
                        st.rerun()
                
                if cancel_dislike:
                    st.session_state[f'show_dislike_form_fuzzy_{i}'] = False
                    st.rerun()

# ===================== UI TABS =====================

def single_sku_tab(df, df_rules, vectorizer, tfidf_matrix):
    """Single SKU processing tab with feedback functionality"""
    st.header("Enter SKU Information")
    
    # Store results in session state to persist after feedback clicks
    if 'search_results' not in st.session_state:
        st.session_state.search_results = None
    if 'search_inputs' not in st.session_state:
        st.session_state.search_inputs = {'sku': '', 'name': ''}
    
    with st.form("sku_search_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            sku_input = st.text_input(
                "SKU Number",
                placeholder="e.g., SV50139.06",
                help="Enter the SKU number (partial matches allowed)"
            )
        
        with col2:
            name_input = st.text_input(
                "SKU Name",
                placeholder="e.g., PKG MATL| COLLAPSIBLE BIN",
                help="Enter the SKU name (partial matches allowed)"
            )
        
        submitted = st.form_submit_button("🔍 Classify SKU", type="primary", use_container_width=True)
    
    if submitted:
        if not sku_input.strip() and not name_input.strip():
            st.warning("Please enter at least one field (SKU Number or SKU Name)")
            return
        
        # Store inputs in session state
        st.session_state.search_inputs = {'sku': sku_input, 'name': name_input}
        
        with st.spinner("Analyzing SKU patterns..."):
            exact_matches = get_exact_predictions(df, sku_input, name_input)
            fuzzy_matches = get_fuzzy_predictions(df, sku_input, name_input, vectorizer, tfidf_matrix, top_k=3)
        
        # Store results in session state
        st.session_state.search_results = {
            'exact_matches': exact_matches,
            'fuzzy_matches': fuzzy_matches
        }
    
    # Display results if they exist
    if st.session_state.search_results is not None:
        exact_matches = st.session_state.search_results['exact_matches']
        fuzzy_matches = st.session_state.search_results['fuzzy_matches']
        sku_input = st.session_state.search_inputs['sku']
        name_input = st.session_state.search_inputs['name']
        
        st.markdown("---")
        st.header("Classification Results")
        
        if not exact_matches.empty:
            display_exact_matches(exact_matches, sku_input, name_input, df)
        else:
            st.info("No exact matches found. Showing fuzzy predictions below.")
        
        display_fuzzy_matches(fuzzy_matches, sku_input, name_input, df_rules)

def bulk_processing_tab(df, vectorizer, tfidf_matrix):
    """Optimized bulk SKU processing tab"""
    st.header("🚀 Bulk SKU Classification")
    st.info("Upload a CSV or Excel file with 'sku number' and 'sku name' columns")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="File should contain columns: sku name, sku number"
    )
    
    if uploaded_file is not None:
        try:
            # Read file based on extension
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                input_df = pd.read_csv(uploaded_file)
            elif file_extension in ['xlsx', 'xls']:
                input_df = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file type. Please upload a CSV or Excel file.")
                return
            
            st.success(f"File uploaded successfully! Found {len(input_df)} rows.")
            
            # Display preview
            with st.expander("📋 File Preview", expanded=False):
                safe_dataframe_display(input_df.head(10), key="file_preview")
            
            # Check required columns
            required_cols = ['sku number', 'sku name']
            input_columns_lower = [col.lower() for col in input_df.columns]
            missing_cols = [col for col in required_cols if col.lower() not in input_columns_lower]
            
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
                st.info("Please ensure your file contains 'sku number' and 'sku name' columns")
                return
            
            # Normalize column names
            column_mapping = {}
            for col in input_df.columns:
                if col.lower() == 'sku number':
                    column_mapping[col] = 'sku number'
                elif col.lower() == 'sku name':
                    column_mapping[col] = 'sku name'
            
            input_df = input_df.rename(columns=column_mapping)
            
            # Show processing info
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rows to Process", len(input_df))
            with col2:
                est_time = len(input_df) * 0.01
                st.metric("Estimated Time", f"{est_time:.1f}s")
            
            # Process button
            if st.button("🚀 Process Classifications", type="primary", use_container_width=True):
                start_time = pd.Timestamp.now()
                
                # Process data
                result_df = ultra_fast_bulk_predictions(input_df, df, vectorizer, tfidf_matrix)
                
                # Apply valid product line filter (same as training)
                valid_cmr_product_lines = [
                    'BEAService', 'BEAHardware', 'BEAOther', 'HardwareConsumables', 
                    'SUTAutomation','2DBioProcessContainers', '3DBioProcessContainers', 
                    'FillFinish', 'FlexibleOther','FluidTransferAssemblies', 
                    'BioproductionContainments', 'BottleAssemblies',
                    'ProductionCellCulture', 'RigidOther', 'SUDOther'
                ]
                result_df = result_df[result_df['Prediction 1: CMR Product Line'].isin(valid_cmr_product_lines)]

                # Exclude SKUs with 'unknown' or 'unknown PROD' in SKU name or number
                exclude_patterns = ['unknown', 'unknown prod']
                for pattern in exclude_patterns:
                    result_df = result_df[~result_df['sku name'].str.contains(pattern, case=False, na=False)]
                    result_df = result_df[~result_df['sku number'].str.contains(pattern, case=False, na=False)]
                
                # Filter out rows with no matches
                result_df = result_df[result_df.filter(like='Prediction').ne('No Match Found').any(axis=1)]
                
                end_time = pd.Timestamp.now()
                processing_time = (end_time - start_time).total_seconds()
                
                st.success(f"Processing completed in {processing_time:.2f} seconds!")
                
                # Performance metrics
                st.subheader("📊 Performance Metrics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Processing Time", f"{processing_time:.2f}s")
                with col2:
                    rows_per_sec = len(input_df) / processing_time if processing_time > 0 else 0
                    st.metric("Rows/Second", f"{rows_per_sec:.1f}")
                with col3:
                    successful_matches = len(result_df[result_df['Prediction 1: SKU Number'] != 'No Match Found'])
                    st.metric("Successful Matches", successful_matches)
                with col4:
                    success_rate = (successful_matches / len(result_df) * 100) if len(result_df) > 0 else 0
                    st.metric("Success Rate", f"{success_rate:.1f}%")
                
                # Results preview
                st.subheader("🔍 Results Preview")
                prediction_cols = [col for col in result_df.columns if col.startswith('Prediction 1:')]
                preview_cols = ['sku number', 'sku name'] + prediction_cols
                safe_dataframe_display(result_df[preview_cols].head(10), key="results_preview")
                
                # Download section
                st.subheader("💾 Download Results")
                timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                
                csv_data = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv_data,
                    file_name=f"SKU_Classifications_{timestamp}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# ===================== MAIN APPLICATION =====================

def analytics_dashboard():
    """Display analytics dashboard"""
    st.header("📈 Feedback Analytics Dashboard")
    
    feedback_df = load_feedback_data()
    
    if feedback_df.empty:
        st.info("No feedback data available yet")
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_feedback = len(feedback_df)
    likes = len(feedback_df[feedback_df['feedback'] == 'like'])
    dislikes = len(feedback_df[feedback_df['feedback'] == 'dislike'])
    accuracy = (likes / total_feedback * 100) if total_feedback > 0 else 0
    
    with col1:
        st.metric("Total Feedback", total_feedback)
    with col2:
        st.metric("👍 Likes", likes)
    with col3:
        st.metric("👎 Dislikes", dislikes)
    with col4:
        st.metric("Accuracy Rate", f"{accuracy:.1f}%")
    
    # Feedback by user
    st.subheader("Feedback by User")
    user_feedback = feedback_df.groupby(['user_email', 'feedback']).size().unstack(fill_value=0)
    st.bar_chart(user_feedback)
    
    # Recent feedback
    st.subheader("Recent Feedback (Last 10)")
    recent_feedback = feedback_df.tail(10)[['timestamp', 'user_email', 'sku_input', 'predicted_product_line_code', 'confidence_score', 'feedback']]
    st.dataframe(recent_feedback, use_container_width=True)
    
    # Download feedback data
    if st.button("📅 Download Feedback Data", use_container_width=True):
        csv_data = feedback_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name=f"bulk_feedback_data_{time.strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

def main():
    st.set_page_config(
        page_title="SKU Classification System",
        page_icon="🤖",
        layout="wide"
    )

    # Custom CSS for better visuals (removed blue backgrounds)
    st.markdown("""
        <style>
        .main {background-color: #fff;}
        .stTabs [role="tab"] {font-size: 18px;}
        .stApp {background-color: #fff;}
        .stSidebar {background-color: #fff;}
        .metric-label {font-size: 16px;}
        .metric-value {font-size: 22px; font-weight: bold;}
        </style>
    """, unsafe_allow_html=True)

    # Welcome message
    st.markdown("<h1 style='color:#D70000;'>🤖 SKU Classification System</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:#444;'>Welcome to the Thermo Fisher SKU AI Classifier!</h4>", unsafe_allow_html=True)
    st.info("Quick Start: Use the tabs below to classify a single SKU, process bulk files, or view analytics.")

    # Initialize session state
    init_session_state()

    # Authentication
    if not st.session_state.authenticated:
        login_page()
        return

    # Header with logo and user info
    col1, col2, col3 = st.columns([1, 4, 1])
    with col1:
        try:
            st.image("logo2.png", width=1000)
        except:
            st.write("🤖")
    with col2:
        st.markdown(f"<h2 style='color:#D70000;'>Welcome, {st.session_state.user_name}</h2>", unsafe_allow_html=True)
    with col3:
        if st.button("🚪 Logout", use_container_width=True, help="Log out and return to login page"):
            st.session_state.authenticated = False
            st.session_state.user_email = None
            st.session_state.user_name = None
            st.rerun()

    st.markdown("---")

    # Load data
    if 'df' not in st.session_state:
        with st.spinner("Loading training data..."):
            st.session_state.df = load_data()
            if st.session_state.df is not None:
                st.session_state.vectorizer, st.session_state.tfidf_matrix = create_similarity_index(st.session_state.df)

    if 'df_rules' not in st.session_state:
        st.session_state.df_rules = load_business_rules_data()
    
    if 'df_hierarchy' not in st.session_state:
        st.session_state.df_hierarchy = load_reference_hierarchy()

    df = st.session_state.df
    df_rules = st.session_state.df_rules

    if df is None:
        st.error("❌ Failed to load data. Please check if the Training_Set.xlsx file exists.")
        return

    st.success(f'✅ Model ready! Trained on {len(df)} Unique SKU patterns')

    # Tabs with icons
    tab1, tab2, tab3 = st.tabs([
        "🔍 Single SKU Classification", 
        "📂 Bulk Classification", 
        "📈 Analytics Dashboard"
    ])

    with tab1:
        st.markdown("<h3 style='color:#0076D7;'>🔍 Single SKU Classification</h3>", unsafe_allow_html=True)
        st.caption("Enter SKU details below to get instant classification and feedback options.")
        single_sku_tab(df, df_rules, st.session_state.vectorizer, st.session_state.tfidf_matrix)

    with tab2:
        st.markdown("<h3 style='color:#0076D7;'>📂 Bulk Classification</h3>", unsafe_allow_html=True)
        st.caption("Upload a file and classify hundreds of SKUs in seconds. Download results instantly.")
        bulk_processing_tab(df, st.session_state.vectorizer, st.session_state.tfidf_matrix)

    with tab3:
        st.markdown("<h3 style='color:#0076D7;'>📈 Analytics Dashboard</h3>", unsafe_allow_html=True)
        st.caption("View feedback analytics and model performance.")
        analytics_dashboard()

    # Sidebar improvements
    # Remove or comment out this block to remove the sidebar
    # with st.sidebar:
    #     #st.image("logo2.png", width=100)
    #     st.header("📊 SKU Information")
    #     st.metric("Trained on Unique SKUs of", len(df))
    #     st.metric("Product Line Codes", 128)
    #     st.metric("CMR Product Lines", df['cmr product line'].nunique())
    #     st.markdown("---")
    #     st.header("👤 User Info")
    #     st.info(f"**Logged in as:**\n{st.session_state.user_name}\n{st.session_state.user_email}")
    #     st.markdown("---")
    #     st.header("📈 Quick Feedback Stats")
    #     feedback_df = load_feedback_data()
    #     if not feedback_df.empty:
    #         total_feedback = len(feedback_df)
    #         likes = len(feedback_df[feedback_df['feedback'] == 'like'])
    #         dislikes = len(feedback_df[feedback_df['feedback'] == 'dislike'])
    #         st.metric("Total Feedback", total_feedback)
    #         if total_feedback > 0:
    #             st.metric("Accuracy Rate", f"{(likes/total_feedback)*100:.1f}%")
    #             st.metric("👍 Likes", likes)
    #             st.metric("👎 Dislikes", dislikes)
    #     else:
    #         st.info("No feedback data available yet")
    #     admin_users = ['sample@thermofisher.com', 'sample_v1@thermofisher.com']
    #     if st.session_state.user_email in admin_users:
    #         st.markdown("---")
    #         st.header("🔄 Model Improvement")
    #         if st.button("Retrain with Feedback", use_container_width=True, help="Incorporate positive feedback into model training"):
    #             with st.spinner("Retraining model with feedback data..."):
    #                 enhanced_df = retrain_with_feedback()
    #                 if enhanced_df is not None:
    #                     st.session_state.df = enhanced_df
    #                     st.session_state.vectorizer, st.session_state.tfidf_matrix = create_similarity_index(enhanced_df)
    #                     st.success("Model retrained successfully with feedback data!")
    #                     st.rerun()
    #     st.markdown("---")
    #     st.header("📅 Resources")
    #     try:
    #         with open('Business_Rule.xlsx', 'rb') as file:
    #             business_rule_data = file.read()
    #             st.download_button(
    #                 label="Download Business Rules",
    #                 data=business_rule_data,
    #                 file_name="Business_Rule.xlsx",
    #                 mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    #                 use_container_width=True
    #             )
    #     except:
    #         pass

if __name__ == "__main__":
    main()