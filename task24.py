import spacy
import pandas as pd
from typing import List, Tuple, Optional
import warnings
import re
warnings.filterwarnings("ignore", message=".*CUDA path could not be detected.*")

# Global variable for the NLP model
nlp = None

INVESTMENT_BANKS = [
    "Goldman Sachs", "Goldman Sachs Group", "GS",
    "JP Morgan", "JPMorgan", "JPMorgan Chase", "JPM",
    "Bank of America", "BAC",
    "Citigroup", "Citi", "Citibank", "C",
    "Morgan Stanley", "MS",
    "Barclays", "Barclays Bank",
    "Credit Suisse",
    "Deutsche Bank",
    "UBS",
    "Wells Fargo", "WFC",
    "HSBC",
    "BNP Paribas",
    "Societe Generale",
    "Nomura",
    "RBC", "Royal Bank of Canada",
    "Jefferies",
    "Lazard",
    "Evercore",
    "Macquarie",
    "Raymond James"
]

# Enhanced company names with 70 stocks - both ticker symbols and company names

NON_BANK_COMPANIES = [
    # 70 Stock Companies - Ticker Symbols
    "AAPL", "Apple Inc.",
    "AMZN", "Amazon.com Inc.",
    "TSLA", "Tesla Inc.",
    "CAT", "Caterpillar Inc.",
    "UBER", "Uber Technologies Inc.",
    "JPM", "JPMorgan Chase & Co.",
    "PFE", "Pfizer Inc.",
    "PG", "The Procter & Gamble Company",
    "MAR", "Marriott International Inc.",
    "XOM", "Exxon Mobil Corporation",
    "GME", "GameStop Corp.",
    "GOOGL", "Alphabet Inc.",
    "META", "Meta Platforms Inc.",
    "MSFT", "Microsoft Corporation",
    "BA", "The Boeing Company",
    "AMD", "Advanced Micro Devices Inc.",
    "NFLX", "Netflix Inc.",
    "NVDA", "NVIDIA Corporation",
    "BABA", "Alibaba Group Holding Limited",
    "XYZ", "Not a valid/recognized ticker",
    "WMT", "Walmart Inc.",
    "DIS", "The Walt Disney Company",
    "F", "Ford Motor Company",
    "GS", "The Goldman Sachs Group Inc.",
    "NVAX", "Novavax Inc.",
    "JNJ", "Johnson & Johnson",
    "V", "Visa Inc.",
    "PYPL", "PayPal Holdings Inc.",
    "BAC", "Bank of America Corporation",
    "INTC", "Intel Corporation",
    "APP", "Applovin Corporation",
    "ASTS", "AST SpaceMobile Inc.",
    "AVGO", "Broadcom Inc.",
    "CELH", "Celsius Holdings Inc.",
    "COIN", "Coinbase Global Inc.",
    "COST", "Costco Wholesale Corporation",
    "CRM", "Salesforce Inc.",
    "CRWD", "CrowdStrike Holdings Inc.",
    "CVS", "CVS Health Corporation",
    "DELL", "Dell Technologies Inc.",
    "DKNG", "DraftKings Inc.",
    "HDB", "HDFC Bank Limited",
    "HOOD", "Robinhood Markets Inc.",
    "IBN", "ICICI Bank Limited",
    "INFY", "Infosys Limited",
    "LCID", "Lucid Group Inc.",
    "LULU", "Lululemon Athletica Inc.",
    "MA", "Mastercard Incorporated",
    "MMYT", "MakeMyTrip Limited",
    "MRNA", "Moderna Inc.",
    "MSTR", "MicroStrategy Incorporated",
    "MU", "Micron Technology Inc.",
    "NKE", "NIKE Inc.",
    "NU", "Nu Holdings Ltd.",
    "PLTR", "Palantir Technologies Inc.",
    "RBLX", "Roblox Corporation",
    "RDDT", "Reddit Inc.",
    "RDY", "Dr. Reddy's Laboratories Limited",
    "RIVN", "Rivian Automotive Inc.",
    "SMCI", "Super Micro Computer Inc.",
    "SNOW", "Snowflake Inc.",
    "SOFI", "SoFi Technologies Inc.",
    "TGT", "Target Corporation",
    "TTM", "Tata Motors Limited",
    "U", "Unity Software Inc.",
    "UNH", "UnitedHealth Group Incorporated",
    "VKTX", "Viking Therapeutics Inc.",
    "WBA", "Walgreens Boots Alliance Inc.",
    "WIT", "Wipro Limited",
    "ZETA", "Zeta Global Holdings Corp.",
    
    # 70 Stock Companies - Company Names
    "Apple Inc", "Amazon.com Inc", "Tesla Inc", "Caterpillar Inc", "Uber Technologies Inc",
    "Pfizer Inc", "Procter & Gamble Co", "Marriott International Inc", "Exxon Mobil Corp",
    "GameStop Corp", "Alphabet Inc", "Meta Platforms Inc", "Microsoft Corporation",
    "Boeing Co", "Advanced Micro Devices Inc", "Netflix Inc", "NVIDIA Corporation",
    "Alibaba Group Holding Ltd", "Walmart Inc", "Walt Disney Co", "Ford Motor Co",
    "Goldman Sachs Group Inc", "Novavax Inc", "Johnson & Johnson", "Visa Inc",
    "PayPal Holdings Inc", "Bank of America Corp", "Intel Corporation",
    "Digital Turbine Inc", "AST SpaceMobile Inc", "Broadcom Inc", "Celsius Holdings Inc",
    "Coinbase Global Inc", "Costco Wholesale Corp", "Salesforce Inc", "CrowdStrike Holdings Inc",
    "CVS Health Corp", "Dell Technologies Inc", "DraftKings Inc", "HDFC Bank Ltd",
    "Robinhood Markets Inc", "ICICI Bank Ltd", "Infosys Ltd", "Lucid Group Inc",
    "Lululemon Athletica Inc", "Mastercard Inc", "MakeMyTrip Ltd", "Moderna Inc",
    "MicroStrategy Inc", "Micron Technology Inc", "Nike Inc", "Nu Holdings Ltd",
    "Palantir Technologies Inc", "Roblox Corp", "Reddit Inc", "Dr. Reddy's Laboratories Ltd",
    "Rivian Automotive Inc", "Super Micro Computer Inc", "Snowflake Inc", "SoFi Technologies Inc",
    "Target Corp", "TTM Technologies Inc", "Unity Software Inc", "UnitedHealth Group Inc",
    "Veritek Global Inc", "Walgreens Boots Alliance Inc", "Wipro Ltd", "Zeta Global Holdings Corp",
    
    # Additional common variations
    "Procter and Gamble", "P&G", "Walt Disney Company", "Disney", "Johnson and Johnson", "J&J",
    "Exxon", "GameStop", "Advanced Micro Devices", "Broadcom Limited", "Costco",
    "Super Micro Computer", "SMCI", "Unity", "UnitedHealth", "Walgreens", "MakeMyTrip Limited",
    "Dr Reddy's", "Dr. Reddy", "Reddy's Labs", "Micron", "PayPal", "CrowdStrike",
    "Robinhood", "DraftKings", "Lululemon", "Palantir", "Roblx", "Snowflake", "SoFi"
]

# Action keywords that indicate regulatory/negative actions (fallback list)
REGULATORY_ACTIONS_FALLBACK = [
    "investigates", "investigate", "sues", "sue", "fines", "fine", "penalizes", "penalize",
    "sanctions", "sanction", "charges", "charge", "prosecutes", "prosecute"
]

def load_spacy_model(model_name: str = "en_core_web_sm"):
    """Load specified spaCy model with improved model selection"""
    global nlp
    
    # Available models in order of preference for accuracy
    available_models = ["en_core_web_trf", "en_core_web_md", "en_core_web_sm"]
    
    # If user specifies a model, try that first
    if model_name:
        try:
            nlp = spacy.load(model_name)
            print(f"✅ Loaded spaCy model: {model_name}")
            return True
        except OSError:
            print(f"❌ Model {model_name} not found.")
            # If it's the transformer model, provide specific installation instructions
            if model_name == "en_core_web_trf":
                print(f"💡 To install the transformer model, run:")
                print(f"   python -m spacy download en_core_web_trf")
                print(f"   Note: This model requires transformers library and is much larger (~500MB)")
            else:
                print(f"   Install with: python -m spacy download {model_name}")
    
    # Try fallback models if specified model failed
    print("🔄 Trying fallback models...")
    for fallback_model in available_models:
        if fallback_model != model_name:  # Skip if we already tried this one
            try:
                nlp = spacy.load(fallback_model)
                print(f"✅ Loaded fallback spaCy model: {fallback_model}")
                return True
            except OSError:
                print(f"❌ Fallback model {fallback_model} not found.")
                continue
    
    print("❌ No spaCy models available. Please install at least en_core_web_sm:")
    print("   python -m spacy download en_core_web_sm")
    return False

def find_banks_in_text(text: str) -> List[str]:
    """Find all investment banks mentioned in the text with improved accuracy"""
    found_banks = []
    text_lower = text.lower()
    
    # Sort banks by length (longest first) to avoid partial matches
    sorted_banks = sorted(INVESTMENT_BANKS, key=len, reverse=True)
    
    for bank in sorted_banks:
        bank_lower = bank.lower()
        
        # For single letter banks (like "C"), require word boundaries
        if len(bank) <= 2:
            import re
            pattern = r'\b' + re.escape(bank_lower) + r'\b'
            if re.search(pattern, text_lower):
                found_banks.append(bank)
        else:
            # For longer bank names, use regular contains check
            if bank_lower in text_lower:
                # Avoid duplicates (e.g., "Goldman Sachs" and "Goldman Sachs Group")
                if not any(bank_lower in existing.lower() and bank != existing for existing in found_banks):
                    found_banks.append(bank)
    
    return found_banks

def extract_subject_object_improved(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Improved subject-object extraction focusing on financial context"""
    if nlp is None:
        raise Exception("spaCy model not loaded. Call load_spacy_model() first.")
    
    doc = nlp(text)
    
    # Find all banks mentioned in text
    banks_in_text = find_banks_in_text(text)
    
    subject = None
    obj = None
    
    # Look for dependency patterns
    for token in doc:
        # Subject patterns
        if token.dep_ in ["nsubj", "nsubjpass"] and not subject:
            # Get the full noun phrase for subject
            subject = get_full_entity(token, doc)
        
        # Object patterns
        if token.dep_ in ["dobj", "pobj", "obj"] and not obj:
            # Get the full noun phrase for object
            obj = get_full_entity(token, doc)
    
    # If we found banks but no clear subject/object, try pattern matching
    if banks_in_text:
        # Common financial news patterns
        if not subject or not obj:
            subject_alt, obj_alt = extract_by_patterns(text, banks_in_text)
            if not subject:
                subject = subject_alt
            if not obj:
                obj = obj_alt
    
    # Additional improvement: If subject/object extraction failed, use bank detection
    if banks_in_text:
        # Pattern: "Bank does something" - bank should be subject
        for bank in banks_in_text:
            if text.lower().startswith(bank.lower()) and not subject:
                subject = bank
            # Pattern: "Something happens to bank" - bank should be object
            elif not obj and any(action in text.lower() for action in ["investigates", "sues", "fines", "upgrades", "downgrades"]):
                # Check if bank comes after action word
                words = text.lower().split()
                bank_words = bank.lower().split()
                for i, word in enumerate(words):
                    if word in ["investigates", "sues", "fines", "upgrades", "downgrades", "raises", "lowers"]:
                        # Look for bank in remaining words
                        remaining = " ".join(words[i+1:])
                        if bank.lower() in remaining:
                            obj = bank
                            break
    
    return subject, obj

def get_full_entity(token, doc):
    """Get the full entity/phrase instead of just one word - improved version"""
    # If it's part of a named entity, get the full entity
    if token.ent_type_:
        # Find the full entity this token belongs to
        for ent in doc.ents:
            if token.i >= ent.start and token.i < ent.end:
                return ent.text
    
    # Otherwise, get the token with its modifiers
    entity_tokens = [token.text]
    
    # Add dependent tokens (adjectives, compounds, etc.)
    for child in token.children:
        if child.dep_ in ["compound", "amod", "nn", "nmod"]:
            entity_tokens.insert(0, child.text)
    
    # Also check for tokens that this token depends on
    if token.head != token and token.head.dep_ in ["compound", "amod"]:
        entity_tokens.insert(0, token.head.text)
    
    return " ".join(entity_tokens)

def extract_by_patterns(text: str, banks_in_text: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """Extract subject/object using financial news patterns - improved"""
    subject = None
    obj = None
    
    # Improved patterns with better regex
    patterns = [
        # Pattern: "Bank action object" (e.g., "Goldman Sachs raises target")
        r'(Goldman Sachs|JP ?Morgan|JPMorgan|Barclays|UBS|Citigroup|Morgan Stanley|Credit Suisse|Deutsche Bank|Wells Fargo)\s+(raises?|lowers?|upgrades?|downgrades?|initiates?|adds?|boosts?|reduces?|increases?|decreases?)\s+(.+?)(?:\s+to\s+|\s+from\s+|$)',
        
        # Pattern: "Entity action Bank" (e.g., "SEC investigates JP Morgan")
        r'(\w+(?:\s+\w+)*)\s+(investigates?|sues?|fines?|penalizes?|sanctions?|charges?|prosecutes?)\s+(Goldman Sachs|JP ?Morgan|JPMorgan|Barclays|UBS|Citigroup|Morgan Stanley|Credit Suisse|Deutsche Bank|Wells Fargo)',
        
        # Pattern: "Entity action Bank" (e.g., "Rating agency upgrades JP Morgan")
        r'(\w+(?:\s+\w+)*)\s+(upgrades?|downgrades?|raises?|lowers?)\s+(Goldman Sachs|JP ?Morgan|JPMorgan|Barclays|UBS|Citigroup|Morgan Stanley|Credit Suisse|Deutsche Bank|Wells Fargo)',
        
        # Pattern: "Bank stock/shares action" (e.g., "Goldman Sachs stock falls")
        r'(Goldman Sachs|JP ?Morgan|JPMorgan|Barclays|UBS|Citigroup|Morgan Stanley|Credit Suisse|Deutsche Bank|Wells Fargo)\s+(stock|shares?)\s+(falls?|rises?|drops?|gains?)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            groups = match.groups()
            
            # Determine which group is the bank
            bank_group = None
            other_group = None
            
            for i, group in enumerate(groups):
                if any(bank.lower() in group.lower() for bank in banks_in_text):
                    bank_group = group.strip()
                elif i == 0:  # First group is usually subject if not a bank
                    other_group = group.strip()
                elif i == 2 and len(groups) > 2:  # Third group might be object
                    if not other_group:
                        other_group = group.strip()
            
            # Assign subject and object based on sentence structure
            if "investigates" in text.lower() or "sues" in text.lower() or "fines" in text.lower():
                # Bank is object in regulatory actions
                subject = other_group
                obj = bank_group
            elif bank_group and other_group:
                # Check if bank comes first in the sentence
                bank_pos = text.lower().find(bank_group.lower()) if bank_group else -1
                other_pos = text.lower().find(other_group.lower()) if other_group else -1
                
                if bank_pos < other_pos:
                    subject = bank_group
                    obj = other_group
                else:
                    subject = other_group
                    obj = bank_group
            
            break
    
    return subject, obj

def map_to_investment_bank(entity: str) -> Optional[str]:
    """Map an entity to a known investment bank with improved accuracy"""
    if entity is None:
        return None
    
    entity_lower = entity.lower().strip()
    
    # Skip very short entities that might match ticker symbols incorrectly
    if len(entity_lower) <= 2:
        # Only allow exact matches for short entities (like "MS", "GS", "C")
        for bank in INVESTMENT_BANKS:
            if bank.lower() == entity_lower:
                return bank
        return None
    
    # Direct mapping for exact matches
    for bank in INVESTMENT_BANKS:
        if bank.lower() == entity_lower:
            return bank
    
    # Partial matching for longer entities
    for bank in INVESTMENT_BANKS:
        # Skip single letter banks for partial matching
        if len(bank) <= 2:
            continue
            
        if bank.lower() in entity_lower or entity_lower in bank.lower():
            return bank
    
    return None

def is_company_name(entity: str) -> bool:
    """Check if entity is a known company name (non-bank) - Enhanced with 70 stocks"""
    if not entity:
        return False
    
    entity_lower = entity.lower().strip()
    
    # Sort companies by length (longest first) to avoid partial matches
    sorted_companies = sorted(NON_BANK_COMPANIES, key=len, reverse=True)
    
    for company in sorted_companies:
        company_lower = company.lower()
        
        # For ticker symbols (short names), require exact match or word boundary
        if len(company) <= 4 and company.isupper():
            # Exact match for ticker symbols
            if entity_lower == company_lower:
                return True
            # Word boundary match for ticker symbols in longer text
            import re
            pattern = r'\b' + re.escape(company_lower) + r'\b'
            if re.search(pattern, entity_lower):
                return True
        else:
            # For company names, use contains check
            if company_lower in entity_lower or entity_lower in company_lower:
                return True
    
    return False

def has_regulatory_action(text: str) -> bool:
    """Check if text contains regulatory/negative action using AI detection with fallback"""
    
    # Try AI detection first
    try:
        # Import OpenAI here to avoid dependency issues if not available
        from openai import OpenAI
        import os
        from dotenv import load_dotenv
        
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        
        if api_key:
            client = OpenAI(api_key=api_key)
            
            prompt = f"""
Analyze this financial news text and determine if it contains regulatory or legal actions against financial institutions (banks, investment firms, etc.).

Text: "{text}"

Look for actions such as:
- Government investigations, lawsuits, or legal proceedings
- Regulatory fines, penalties, or sanctions
- Compliance violations or enforcement actions
- SEC, CFTC, or other regulatory body actions
- Criminal charges or prosecutions
- Settlement agreements for regulatory violations

Respond with EXACTLY:
regulatory_action: YES or NO
confidence: HIGH, MEDIUM, or LOW
"""
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=50
            )
            
            result = response.choices[0].message.content.strip()
            
            # Parse the response
            if "regulatory_action: YES" in result:
                return True
            elif "regulatory_action: NO" in result:
                return False
    
    except Exception as e:
        # If AI detection fails, fall back to rule-based detection
        print(f"AI regulatory detection failed, using fallback: {e}")
        pass
    
    # Fallback to rule-based detection
    text_lower = text.lower()
    return any(action in text_lower for action in REGULATORY_ACTIONS_FALLBACK)

def is_relevant(subject: Optional[str], obj: Optional[str], text: str, use_new_rule: bool = True) -> bool:
    """
    Determine if news is relevant based on improved rules:
    1. Keep items where the bank is an object (receiving action)
    2. Keep items where the bank is subject AND object is not another company (if use_new_rule=True)
    3. Keep items with regulatory actions against banks
    
    Args:
        use_new_rule: If False, only use old rules (bank as object or regulatory actions)
    """
    
    # First, check if there are any banks in the text at all
    banks_in_text = find_banks_in_text(text)
    if not banks_in_text:
        return False
    
    # Map subject and object to banks
    subject_bank = map_to_investment_bank(subject)
    object_bank = map_to_investment_bank(obj)
    
    # Rule 1: Keep if bank is an object (receiving the action)
    if object_bank:
        return True
    
    # Rule 2: Keep if bank is subject AND object is not another company (NEW RULE)
    if use_new_rule and subject_bank:
        # If object is not a known company, keep it
        if not is_company_name(obj):
            return True
        # If it's a regulatory action, always keep it
        if has_regulatory_action(text):
            return True
    
    # Rule 3: Additional check for regulatory actions
    if has_regulatory_action(text) and banks_in_text:
        return True
    
    # Additional check: if subject is not a bank but object contains bank info
    if not subject_bank and obj and any(bank.lower() in obj.lower() for bank in banks_in_text):
        return True
    
    # Special case: if both subject and object are banks (rare but possible)
    if subject_bank and object_bank:
        return True
    
    return False

def calculate_accuracy_score(subject: Optional[str], obj: Optional[str], text: str) -> float:
    """
    Calculate accuracy of subject-object extraction based on manual rules
    Returns score from 0.0 to 1.0 instead of YES/NO
    """
    
    banks_in_text = find_banks_in_text(text)
    
    # Base score
    score = 0.0
    
    # If no banks found but we extracted something, probably wrong
    if not banks_in_text and (subject or obj):
        return 0.1
    
    # If banks found but nothing extracted, probably wrong
    if banks_in_text and not subject and not obj:
        return 0.2
    
    # Check if extracted entities make sense
    subject_bank = map_to_investment_bank(subject)
    object_bank = map_to_investment_bank(obj)
    
    # Good patterns get higher scores:
    
    # Pattern 1: Bank as subject doing actions: "Goldman Sachs raises target"
    if subject_bank and obj:
        action_words = ["raises", "lowers", "upgrades", "downgrades", "initiates", "adds", "boosts", "target", "price", "rating", "coverage"]
        if any(word in text.lower() for word in action_words):
            score += 0.5
        else:
            score += 0.3
    
    # Pattern 2: External entity acting on bank: "SEC investigates JP Morgan"
    if not subject_bank and object_bank:
        regulatory_entities = ["sec", "regulators", "investors", "rating agency", "agency"]
        if subject and any(entity in subject.lower() for entity in regulatory_entities):
            score += 0.5
        else:
            score += 0.4
    
    # Pattern 3: Bank stock/shares movements
    if subject_bank and ("stock" in text.lower() or "shares" in text.lower()):
        score += 0.4
    
    # Bonus points for having both subject and object extracted
    if subject and obj:
        score += 0.2
    
    # Bonus for matching banks in text
    if (subject_bank or object_bank) and banks_in_text:
        score += 0.2
    
    # Penalty for completely wrong extraction
    if banks_in_text and not (subject_bank or object_bank):
        score -= 0.1
    
    # Clamp score between 0 and 1
    return max(0.0, min(1.0, score))

def process_csv(input_csv: str, model_name: str = "en_core_web_trf", use_new_rule: bool = True) -> pd.DataFrame:
    """
    Process CSV with improved subject-object extraction and filtering
    
    Args:
        input_csv: Path to input CSV file
        model_name: spaCy model to use ('en_core_web_sm', 'en_core_web_md', 'en_core_web_trf')
        use_new_rule: Whether to use the new filtering rule for banks as subjects
    """
    
    # Load the specified model
    if not load_spacy_model(model_name):
        raise Exception(f"Failed to load model {model_name}")
    
    df = pd.read_csv(input_csv, encoding="cp1252")
    results = []

    print(f"\n🔍 Processing with model: {model_name}")
    print(f"📏 Using new rule: {use_new_rule}")
    print(f"📈 Enhanced with 70 stock companies for better filtering")
    print(f"🤖 AI-powered regulatory action detection enabled")
    print("-" * 60)

    for i, row in df.iterrows():
        text = row["text"]
        print(f"\n[{i+1}/{len(df)}] Processing: {text[:60]}...")
        
        # Extract subject and object
        subj, obj = extract_subject_object_improved(text)
        
        # Map to banks
        subj_bank = map_to_investment_bank(subj)
        obj_bank = map_to_investment_bank(obj)
        
        # Determine relevance
        relevance = is_relevant(subj, obj, text, use_new_rule)
        
        # Calculate accuracy score (0.0 to 1.0)
        accuracy_score = calculate_accuracy_score(subj, obj, text)
        
        print(f"  Subject: '{subj}' -> Bank: {subj_bank}")
        print(f"  Object: '{obj}' -> Bank: {obj_bank}")
        print(f"  Relevant: {relevance} | Accuracy Score: {accuracy_score:.2f}")
        
        results.append({
            "text": text,
            "subject": subj_bank if subj_bank else subj,
            "object": obj_bank if obj_bank else obj,
            "relevant": "YES" if relevance else "NO",
            "accuracy": accuracy_score,  # Now a float from 0.0 to 1.0
            "model_used": model_name,
            "new_rule_used": "YES" if use_new_rule else "NO"
        })

    result_df = pd.DataFrame(results)
    
    # Print summary statistics
    total = len(result_df)
    relevant = len(result_df[result_df['relevant'] == 'YES'])
    avg_accuracy = result_df['accuracy'].mean()
    high_accuracy = len(result_df[result_df['accuracy'] >= 0.7])
    
    print(f"\n📊 Summary:")
    print(f"   Total items: {total}")
    print(f"   Relevant: {relevant} ({relevant/total*100:.1f}%)")
    print(f"   Average accuracy score: {avg_accuracy:.3f}")
    print(f"   High accuracy (≥0.7): {high_accuracy} ({high_accuracy/total*100:.1f}%)")
    print(f"   📈 Enhanced filtering with 70 stock companies")
    
    return result_df