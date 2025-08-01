# SKU Classification Bot

A Streamlit-based AI system for classifying SKU (Stock Keeping Unit) product lines using machine learning and fuzzy matching algorithms.

## Features

### 🔐 Authentication System
- ThermoFisher email domain validation (@thermofisher.com)
- Microsoft Teams integration for login notifications

### 🔍 Single SKU Classification
- Real-time SKU classification with exact and fuzzy matching
- TF-IDF vectorization for similarity analysis
- Business rule integration for pattern recognition
- Interactive feedback system with like/dislike functionality

### 📂 Bulk Processing
- High-performance batch processing for CSV/Excel files
- Optimized algorithms processing hundreds of SKUs in seconds
- Downloadable results with confidence scores
- Support for multiple prediction candidates per SKU

### 📈 Analytics Dashboard
- User feedback analytics and performance metrics
- Model accuracy tracking
- Downloadable feedback data for analysis

### 🤖 Smart Classification Engine
- CMR product line correction algorithms
- Cosine similarity matching with configurable thresholds
- Business rule pattern matching

## Installation

### Prerequisites
```bash
pip install streamlit pandas numpy scikit-learn openpyxl requests
```

### Required Files
- `Training_Set.xlsx` - Training data with SKU patterns
- `Business_Rule.xlsx` - Business rules for pattern matching
- `teams_config.py` - Microsoft Teams webhook configuration
- `logo2.png` - Application logo (optional)

## Usage

### Starting the Application
```bash
streamlit run main.py
```

### Input Data Format
For bulk processing, ensure your CSV/Excel file contains:
- `sku number` - SKU identifier
- `sku name` - SKU description

### Training Data Structure
The training dataset has been included:
- `sku number` - SKU identifier
- `sku name` - SKU description  
- `product line code` - Product classification code
- `cmr product line` - CMR classification
- `product line name` - Product line description
- `sub platform` - Business unit classification

## Core Algorithms

### Volume-Based Classification
- Extracts volume from SKU names using regex patterns
- Applies 50L threshold rule for 2D/3D container classification
- Supports liter (L) and milliliter (ML) units

### Similarity Matching
- **Exact Matching**: Substring containment for high-confidence results
- **Fuzzy Matching**: TF-IDF vectorization with cosine similarity
- **Character N-grams**: 1-3 character patterns for robust matching

### Product Line Mapping
- Dynamic 2D to 3D product line code conversion
- Volume-based CMR product line correction
- Business rule validation and pattern matching

## Configuration

### Valid CMR Product Lines
```python
valid_cmr_product_lines = [
    'BEAService', 'BEAHardware', 'BEAOther', 'HardwareConsumables',
    'SUTAutomation', '2DBioProcessContainers', '3DBioProcessContainers',
    'FillFinish', 'FlexibleOther', 'FluidTransferAssemblies',
    'BioproductionContainments', 'BottleAssemblies',
    'ProductionCellCulture', 'RigidOther', 'SUDOther'
]
```

### Performance Optimization
- LRU caching for similarity calculations
- Vectorized operations for bulk processing
- Session state management for UI persistence

## File Structure
```
sku-classification-bot/
├── main.py                 # Main application
├── Training_Set.xlsx       # Training data
├── Business_Rule.xlsx      # Business rules
├── teams_config.py         # Teams integration
├── logo2.png              # Application logo
├── user_credentials/       # User login data
└── feedback_data/          # User feedback storage
```

## API Integration

### Microsoft Teams Notifications
- Login event notifications
- Feedback submission alerts
- Configurable webhook endpoints

### Feedback System
- JSON and CSV feedback storage
- User correction tracking
- Model improvement suggestions

## Performance Metrics
- Processing speed: ~100 rows/second for bulk operations
- Memory efficient with caching optimization
- Real-time confidence scoring
- Success rate tracking and analytics

## Security Features
- Domain-restricted authentication
- Session state management
- Input validation and sanitization
- Error handling and logging

## Contributing
1. Ensure all dependencies are installed
2. Maintain code formatting standards
3. Test with sample data before deployment
4. Update documentation for new features