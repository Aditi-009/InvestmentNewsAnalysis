import os
from openai import OpenAI
import pandas as pd
from typing import Tuple
import re
from dotenv import load_dotenv

load_dotenv()

# Fix OpenAI API key initialization - use environment variable properly
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")  # Use standard OPENAI_API_KEY environment variable
)

INVESTMENT_BANKS = [
    "Goldman Sachs", "Goldman Sachs Group", "GS",
    "JP Morgan", "JPMorgan", "JPMorgan Chase", "JPM",
    "Bank of America", "BAC",
    "Citigroup", "Citi", "Citibank", "C",
    "Morgan Stanley", "MS",
    "Barclays", "Credit Suisse", "Deutsche Bank", "UBS",
    "Wells Fargo", "HSBC", "BNP Paribas", "Societe Generale", "Nomura",
    "RBC", "Royal Bank of Canada", "Jefferies", "Lazard",
    "Evercore", "Macquarie", "Raymond James"
]

def extract_bank_from_text(text: str) -> str:
    """Extract investment bank from text using keyword matching"""
    text_lower = text.lower()
    for bank in INVESTMENT_BANKS:
        if bank.lower() in text_lower:
            return bank
    return "None"

def analyze_sentiment_simple(text: str) -> Tuple[float, str]:
    """Simple rule-based sentiment analysis as fallback with -1 to +1 scale"""
    text_lower = text.lower()
    
    # Positive keywords
    positive_words = ['upgrades', 'raises', 'buy', 'boosts', 'increases', 'initiates coverage', 
                     'overweight', 'conviction buy', 'top picks', 'strong', 'surge', 'positive',
                     'outperform', 'bullish', 'recommend']
    
    # Negative keywords  
    negative_words = ['downgrades', 'lowers', 'falls', 'fine', 'investigates', 'sue', 
                     'decreased', 'reduces', 'drops', 'cuts', 'negative', 'bearish',
                     'underperform', 'sells', 'decline']
    
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    if positive_count > negative_count:
        score = 0.3 + (positive_count * 0.2)  # Range: 0.3 to 1.0
        sentiment = "Positive"
    elif negative_count > positive_count:
        score = -0.3 - (negative_count * 0.2)  # Range: -1.0 to -0.3
        sentiment = "Negative"
    else:
        score = 0.0
        sentiment = "Neutral"
    
    # Clamp score between -1 and 1
    score = max(-1.0, min(1.0, score))
    
    return score, sentiment

def call_gpt_for_sentiment(text: str) -> Tuple[float, str, str]:
    """Call GPT for sentiment analysis with robust error handling and target sentiment"""
    
    # First, extract the bank name from the text
    target_bank = extract_bank_from_text(text)
    
    prompt = f"""
Analyze this financial news text and provide sentiment specifically about the mentioned investment bank.

Investment Banks: {', '.join(INVESTMENT_BANKS)}

Text: "{text}"

Provide:
1. Target Sentiment Score: A score from -1.0 (very negative) to +1.0 (very positive) for the investment bank mentioned, with 0.0 being neutral
2. Target Sentiment Label: Positive, Negative, or Neutral for the investment bank
3. Target Bank: The specific investment bank mentioned in the text (or 'None' if no bank is mentioned)

Example:
Text: "Goldman Sachs upgrades Apple to Buy with $200 target"
Target Sentiment Score: 0.0 (this news is about Goldman Sachs making a recommendation, neutral for Goldman Sachs itself)
Target Sentiment Label: Neutral
Target Bank: Goldman Sachs

Text: "SEC investigates Goldman Sachs for trading violations"
Target Sentiment Score: -0.8 (negative news directly affecting Goldman Sachs)
Target Sentiment Label: Negative  
Target Bank: Goldman Sachs

Format your response EXACTLY as:
target_sentiment_score: X.X
target_sentiment_label: [Positive/Negative/Neutral]
target_bank: [bank name or None]
"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=150
        )
        
        reply = response.choices[0].message.content.strip()
        print(f"GPT Response: {reply}")
        
        # Parse the response more robustly
        lines = [line.strip() for line in reply.split('\n') if line.strip()]
        
        score = 0.0
        sentiment = "Neutral" 
        target = target_bank if target_bank != "None" else "None"
        
        for line in lines:
            if line.startswith('target_sentiment_score:'):
                try:
                    score_match = re.findall(r'target_sentiment_score:\s*([-\d.]+)', line)
                    if score_match:
                        score = float(score_match[0])
                        # Ensure score is within -1 to 1 range
                        score = max(-1.0, min(1.0, score))
                except:
                    pass
            elif line.startswith('target_sentiment_label:'):
                sentiment_match = re.findall(r'target_sentiment_label:\s*(\w+)', line)
                if sentiment_match:
                    sentiment = sentiment_match[0]
            elif line.startswith('target_bank:'):
                target_match = re.findall(r'target_bank:\s*(.+)', line)
                if target_match:
                    target = target_match[0].strip()
        
        return score, sentiment, target
        
    except Exception as e:
        print(f"Error during GPT call: {e}")
        print("Falling back to rule-based sentiment analysis...")
        
        # Fallback to simple rule-based analysis
        score, sentiment = analyze_sentiment_simple(text)
        target = target_bank
        
        return score, sentiment, target

def process_csv_with_llm(df: pd.DataFrame) -> pd.DataFrame:
    """Process CSV with LLM sentiment analysis - ONLY for relevant items and exclude non-relevant from output"""
    
    # Count relevant vs non-relevant items
    total_items = len(df)
    relevant_df = df[df['relevant'] == 'YES'].copy()
    non_relevant_count = len(df[df['relevant'] == 'NO'])
    
    print(f"\n💭 Processing sentiment analysis and filtering:")
    print(f"   Total items: {total_items}")
    print(f"   Relevant items (will process and include): {len(relevant_df)}")
    print(f"   Non-relevant items (will exclude from output): {non_relevant_count}")
    print("-" * 60)

    if len(relevant_df) == 0:
        print("⚠️ No relevant items found. Returning empty DataFrame with expected columns.")
        # Return empty DataFrame with all expected columns
        empty_df = pd.DataFrame(columns=[
            'text', 'subject', 'object', 'relevant', 'accuracy_AP', 'Accuracy SA',
            'target_sentiment_score', 'target_sentiment', 'target_bank',
            'model_used', 'new_rule_used'
        ])
        return empty_df

    target_sentiment_scores = []
    target_sentiments = []
    target_banks = []

    # Process only relevant items
    for idx, (original_idx, row) in enumerate(relevant_df.iterrows()):
        print(f"\n📰 Processing relevant item {idx+1}/{len(relevant_df)} (original row {original_idx+1}):")
        print(f"Text: {row['text']}")
        
        print("✅ Item is relevant - processing with LLM...")
        score, sentiment, target = call_gpt_for_sentiment(row['text'])
        print(f"✅ Target Sentiment Score: {score}, Target Sentiment: {sentiment}, Target Bank: {target}")

        target_sentiment_scores.append(score)
        target_sentiments.append(sentiment)
        target_banks.append(target)

    # Add sentiment columns to relevant items only
    relevant_df["target_sentiment_score"] = target_sentiment_scores
    relevant_df["target_sentiment"] = target_sentiments
    relevant_df["target_bank"] = target_banks

    # Print processing summary
    print(f"\n📊 Processing Summary:")
    print(f"   Total original items: {total_items}")
    print(f"   Items processed with LLM: {len(relevant_df)}")
    print(f"   Items excluded (not relevant): {non_relevant_count}")
    print(f"   Final output items: {len(relevant_df)}")

    # Reorder columns to include all original columns plus new sentiment columns
    columns = list(relevant_df.columns)
    # Move sentiment columns to the end
    sentiment_cols = ["target_sentiment_score", "target_sentiment", "target_bank"]
    other_cols = [col for col in columns if col not in sentiment_cols]
    final_df = relevant_df[other_cols + sentiment_cols]

    return final_df

def load_csv_with_encoding(filepath: str) -> pd.DataFrame:
    """Load CSV with automatic encoding detection"""
    encodings_to_try = ['utf-8', 'cp1252', 'iso-8859-1', 'latin-1']
    
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(filepath, encoding=encoding)
            print(f"✅ Successfully loaded CSV with {encoding} encoding")
            return df
        except UnicodeDecodeError:
            print(f"❌ Failed with {encoding} encoding, trying next...")
            continue
        except Exception as e:
            print(f"❌ Error with {encoding}: {e}")
            continue
    
    raise Exception(f"Could not read CSV file {filepath} with any supported encoding")

def add_manual_accuracy_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add a manual accuracy column for AI-based accuracy assessment with numerical scores"""
    
    def assess_accuracy_with_ai(text: str, subject: str, obj: str, relevant: str) -> float:
        """Use AI to assess the accuracy of subject-object extraction with numerical score"""
        
        prompt = f"""
Analyze this financial news text and determine the accuracy of the subject-object extraction on a scale from 0.0 to 1.0.

Text: "{text}"
Extracted Subject: "{subject}"
Extracted Object: "{obj}"
Marked as Relevant: {relevant}

Rate the accuracy considering:
1. How well the subject identifies the main entity performing an action (0.0-0.4 points)
2. How well the object identifies the entity receiving the action (0.0-0.4 points)
3. Overall coherence and relevance of the extraction (0.0-0.2 points)

Examples:
- "Goldman Sachs raises Apple target" → Subject: Goldman Sachs, Object: Apple target → Score: 0.9 (excellent)
- "SEC investigates JP Morgan" → Subject: SEC, Object: JP Morgan → Score: 1.0 (perfect)
- "Goldman Sachs stock falls" → Subject: Goldman Sachs, Object: stock → Score: 0.8 (good)
- Wrong extraction or missing key entities → Score: 0.1-0.3 (poor)

Respond with EXACTLY this format:
accuracy_score: X.X
explanation: [brief reason for the score]
"""
        
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=100
            )
            
            result = response.choices[0].message.content.strip()
            
            # Extract accuracy score
            score_match = re.search(r'accuracy_score:\s*([\d.]+)', result)
            if score_match:
                score = float(score_match.group(1))
                return max(0.0, min(1.0, score))  # Clamp between 0 and 1
            else:
                return 0.5  # Default middle score if parsing fails
            
        except Exception as e:
            print(f"Error in AI accuracy assessment: {e}")
            # Fallback to rule-based assessment
            if subject and obj and any(bank.lower() in text.lower() for bank in INVESTMENT_BANKS):
                return 0.7  # Good score for basic requirements met
            return 0.3  # Lower score for poor extraction
    
    print("\n🤖 Adding AI-based accuracy assessment with numerical scores...")
    accuracy_scores = []
    
    for idx, row in df.iterrows():
        print(f"Assessing accuracy for row {idx+1}...")
        accuracy = assess_accuracy_with_ai(
            str(row.get('text', '')),
            str(row.get('subject', '')),
            str(row.get('object', '')),
            str(row.get('relevant', ''))
        )
        accuracy_scores.append(accuracy)
        print(f"  Accuracy Score: {accuracy}")
    
    # Insert the accuracy column after 'relevant' column
    if 'relevant' in df.columns:
        relevant_idx = df.columns.get_loc('relevant')
        df.insert(relevant_idx + 1, 'Accuracy SA', accuracy_scores)
    else:
        df['Accuracy SA'] = accuracy_scores
    
    return df
