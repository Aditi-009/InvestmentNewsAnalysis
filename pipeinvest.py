from task24 import process_csv as relevance_filter
from bankllm import process_csv_with_llm, add_manual_accuracy_column, load_csv_with_encoding
import pandas as pd
import time
from datetime import datetime

def run_complete_pipeline_with_accuracy(input_csv: str, model_name: str = "en_core_web_trf", 
                                       use_new_rule: bool = True, output_csv: str = None):
    """
    Run complete pipeline with accuracy assessment and updated sentiment analysis
    
    Args:
        input_csv: Input CSV file path
        model_name: spaCy model to use (default: en_core_web_trf for best accuracy)
        use_new_rule: Whether to use new filtering rule
        output_csv: Output file path (auto-generated if None)
    """
    
    if output_csv is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rule_suffix = "new_rule" if use_new_rule else "old_rule"
        output_csv = f"SA_Accuracy_Scores_results_{model_name}_{rule_suffix}_{timestamp}.csv"
    
    print(f"🚀 Running Complete Pipeline with Numerical Accuracy Scores")
    print(f"📖 Input: {input_csv}")
    print(f"🤖 Model: {model_name}")
    print(f"📏 New Rule: {'Enabled' if use_new_rule else 'Disabled'}")
    print(f"📝 Output: {output_csv}")
    print("-" * 80)
    
    try:
        # Stage 1: Relevance filtering with accuracy scoring
        print("🔍 Stage 1: Filtering relevant news with accuracy scoring...")
        filtered_df = relevance_filter(input_csv, model_name, use_new_rule)
        
        relevant_count = len(filtered_df[filtered_df['relevant'] == 'YES'])
        total_count = len(filtered_df)
        avg_accuracy = filtered_df['accuracy'].mean()
        high_accuracy_count = len(filtered_df[filtered_df['accuracy'] >= 0.7])
        
        print(f"✅ Stage 1 complete:")
        print(f"   Relevant: {relevant_count}/{total_count} ({relevant_count/total_count*100:.1f}%)")
        print(f"   Average accuracy (rule-based): {avg_accuracy:.3f}")
        print(f"   High accuracy (≥0.7): {high_accuracy_count}/{total_count} ({high_accuracy_count/total_count*100:.1f}%)")
        
        # Stage 2: Add AI-based accuracy assessment
        print("\n🤖 Stage 2: Adding AI-based accuracy assessment with scores...")
        result_df = add_manual_accuracy_column(filtered_df)
        
        ai_avg_accuracy = result_df['Accuracy SA'].mean()
        ai_high_accuracy_count = len(result_df[result_df['Accuracy SA'] >= 0.7])
        print(f"✅ Stage 2 complete:")
        print(f"   AI Average Accuracy: {ai_avg_accuracy:.3f}")
        print(f"   AI High Accuracy (≥0.7): {ai_high_accuracy_count}/{total_count} ({ai_high_accuracy_count/total_count*100:.1f}%)")
        
        # Rename the original accuracy column for clarity
        if 'accuracy' in result_df.columns:
            result_df = result_df.rename(columns={'accuracy': 'accuracy_AP'})
        
        # Stage 3: Sentiment analysis with updated -1 to +1 scale
        print("\n💭 Stage 3: Analyzing target sentiment (-1 to +1 scale)...")
        final_df = process_csv_with_llm(result_df)
        print("✅ Stage 3 complete: Target sentiment analysis finished")
        
        # Stage 4: Save results with UTF-8 encoding
        print(f"\n💾 Stage 4: Saving results to {output_csv}...")
        final_df.to_csv(output_csv, index=False, encoding='utf-8')
        print("✅ Stage 4 complete: Results saved successfully")
        
        # Final summary
        print(f"\n📊 Final Summary:")
        print(f"   Model Used: {model_name}")
        print(f"   New Rule: {'Yes' if use_new_rule else 'No'}")
        print(f"   Total Items: {total_count}")
        print(f"   Relevant Items: {relevant_count} ({relevant_count/total_count*100:.1f}%)")
        print(f"   Rule-based Avg Accuracy: {avg_accuracy:.3f}")
        print(f"   AI-based Avg Accuracy: {ai_avg_accuracy:.3f}")
        
        # Accuracy score distribution
        if 'accuracy_AP' in final_df.columns:
            rule_accuracy_dist = final_df['accuracy_AP'].describe()
            print(f"   Rule-based Accuracy Distribution:")
            print(f"     Min: {rule_accuracy_dist['min']:.3f}, Max: {rule_accuracy_dist['max']:.3f}")
            print(f"     25%: {rule_accuracy_dist['25%']:.3f}, 50%: {rule_accuracy_dist['50%']:.3f}, 75%: {rule_accuracy_dist['75%']:.3f}")
        
        if 'Accuracy SA' in final_df.columns:
            ai_accuracy_dist = final_df['Accuracy SA'].describe()
            print(f"   AI-based Accuracy Distribution:")
            print(f"     Min: {ai_accuracy_dist['min']:.3f}, Max: {ai_accuracy_dist['max']:.3f}")
            print(f"     25%: {ai_accuracy_dist['25%']:.3f}, 50%: {ai_accuracy_dist['50%']:.3f}, 75%: {ai_accuracy_dist['75%']:.3f}")
        
        # Target sentiment breakdown
        if 'target_sentiment' in final_df.columns:
            sentiment_counts = final_df['target_sentiment'].value_counts()
            print(f"   Target Sentiment Distribution:")
            for sentiment, count in sentiment_counts.items():
                print(f"     {sentiment}: {count} ({count/len(final_df)*100:.1f}%)")
            
            # Show sentiment score statistics
            if 'target_sentiment_score' in final_df.columns:
                scores = final_df['target_sentiment_score']
                print(f"   Target Sentiment Score Stats:")
                print(f"     Mean: {scores.mean():.3f}")
                print(f"     Range: {scores.min():.3f} to {scores.max():.3f}")
                print(f"     Std Dev: {scores.std():.3f}")
        
        print(f"\n✅ Complete pipeline finished successfully!")
        print(f"📄 Results saved to: {output_csv}")
        
        return final_df
        
    except Exception as e:
        print(f"❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_model_comparison_with_accuracy(input_csv: str = "sample_investment_news.csv"):
    """
    Run comprehensive model comparison with accuracy assessment using numerical scores
    """
    
    print("🚀 Starting Comprehensive Model Comparison with Numerical Accuracy Scores")
    print("=" * 80)
    
    # Models to test (prioritize transformer model)
    models_to_test = [
        "en_core_web_trf",  # Transformer model (best accuracy)
        "en_core_web_md",   # Medium model  
        "en_core_web_sm"    # Small model
    ]
    
    # Rule configurations
    rule_configs = [
        (True, "with_new_rule"),
        (False, "without_new_rule")
    ]
    
    all_results = {}
    
    for model in models_to_test:
        print(f"\n🔍 Testing model: {model}")
        print("-" * 60)
        
        # Check if model is available
        try:
            import spacy
            spacy.load(model)
        except OSError:
            print(f"❌ Model {model} not available. Skipping...")
            if model == "en_core_web_trf":
                print(f"💡 To install the transformer model (best accuracy), run:")
                print(f"   python -m spacy download en_core_web_trf")
                print(f"   Note: Requires transformers library and ~500MB download")
            continue
        
        for use_new_rule, rule_name in rule_configs:
            print(f"\n📏 Configuration: {rule_name}")
            
            try:
                result_df = run_complete_pipeline_with_accuracy(
                    input_csv, model, use_new_rule
                )
                
                if result_df is not None:
                    # Calculate comprehensive metrics
                    total_items = len(result_df)
                    relevant_items = len(result_df[result_df['relevant'] == 'YES'])
                    
                    # Accuracy metrics
                    rule_avg_accuracy = result_df['accuracy_AP'].mean() if 'accuracy_AP' in result_df.columns else 0
                    ai_avg_accuracy = result_df['Accuracy SA'].mean() if 'Accuracy SA' in result_df.columns else 0
                    rule_high_accuracy = len(result_df[result_df['accuracy_AP'] >= 0.7]) if 'accuracy_AP' in result_df.columns else 0
                    ai_high_accuracy = len(result_df[result_df['Accuracy SA'] >= 0.7]) if 'Accuracy SA' in result_df.columns else 0
                    
                    # Sentiment statistics
                    sentiment_stats = {}
                    if 'target_sentiment_score' in result_df.columns:
                        scores = result_df['target_sentiment_score']
                        sentiment_stats = {
                            'mean_sentiment': scores.mean(),
                            'sentiment_std': scores.std(),
                            'sentiment_min': scores.min(),
                            'sentiment_max': scores.max(),
                            'positive_count': len(result_df[result_df['target_sentiment'] == 'Positive']),
                            'negative_count': len(result_df[result_df['target_sentiment'] == 'Negative']),
                            'neutral_count': len(result_df[result_df['target_sentiment'] == 'Neutral'])
                        }
                    
                    metrics = {
                        'model': model,
                        'rule_config': rule_name,
                        'total_items': total_items,
                        'relevant_items': relevant_items,
                        'rule_avg_accuracy': rule_avg_accuracy,
                        'ai_avg_accuracy': ai_avg_accuracy,
                        'rule_high_accuracy': rule_high_accuracy,
                        'ai_high_accuracy': ai_high_accuracy,
                        'relevance_rate': relevant_items / total_items * 100,
                        'rule_high_accuracy_rate': rule_high_accuracy / total_items * 100,
                        'ai_high_accuracy_rate': ai_high_accuracy / total_items * 100,
                        **sentiment_stats
                    }
                    
                    all_results[f"{model}_{rule_name}"] = metrics
                    
                    print(f"✅ Completed {model} with {rule_name}")
                    print(f"   Relevant: {relevant_items}/{total_items} ({metrics['relevance_rate']:.1f}%)")
                    print(f"   Rule Avg Accuracy: {rule_avg_accuracy:.3f}")
                    print(f"   AI Avg Accuracy: {ai_avg_accuracy:.3f}")
                    print(f"   Rule High Accuracy: {rule_high_accuracy}/{total_items} ({metrics['rule_high_accuracy_rate']:.1f}%)")
                    print(f"   AI High Accuracy: {ai_high_accuracy}/{total_items} ({metrics['ai_high_accuracy_rate']:.1f}%)")
                
            except Exception as e:
                print(f"❌ Failed to test {model} with {rule_name}: {e}")
                continue
    
    # Generate comparison report
    if all_results:
        comparison_df = pd.DataFrame(list(all_results.values()))
        
        # Save comparison report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"comprehensive_accuracy_scores_comparison_{timestamp}.csv"
        comparison_df.to_csv(report_file, index=False)
        
        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE ACCURACY SCORES COMPARISON REPORT")
        print("=" * 80)
        
        print(f"\n📋 Results Summary:")
        display_cols = ['model', 'rule_config', 'relevance_rate', 'rule_avg_accuracy', 'ai_avg_accuracy', 'rule_high_accuracy_rate', 'ai_high_accuracy_rate']
        if all(col in comparison_df.columns for col in display_cols):
            print(comparison_df[display_cols].round(3).to_string(index=False))
        
        # Find best performers
        if len(comparison_df) > 0:
            print(f"\n🏆 Best Performers:")
            if 'ai_avg_accuracy' in comparison_df.columns:
                best_ai_accuracy = comparison_df.loc[comparison_df['ai_avg_accuracy'].idxmax()]
                print(f"   Best AI Avg Accuracy: {best_ai_accuracy['model']} ({best_ai_accuracy['rule_config']}) - {best_ai_accuracy['ai_avg_accuracy']:.3f}")
            
            if 'rule_avg_accuracy' in comparison_df.columns:
                best_rule_accuracy = comparison_df.loc[comparison_df['rule_avg_accuracy'].idxmax()]
                print(f"   Best Rule Avg Accuracy: {best_rule_accuracy['model']} ({best_rule_accuracy['rule_config']}) - {best_rule_accuracy['rule_avg_accuracy']:.3f}")
            
            if 'relevance_rate' in comparison_df.columns:
                best_relevance = comparison_df.loc[comparison_df['relevance_rate'].idxmax()]
                print(f"   Best Relevance: {best_relevance['model']} ({best_relevance['rule_config']}) - {best_relevance['relevance_rate']:.1f}%")
        
        print(f"\n📄 Detailed comparison saved to: {report_file}")
        return comparison_df
    
    else:
        print("❌ No successful results to compare")
        return None

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--compare":
            # Run full model comparison with accuracy scores
            run_model_comparison_with_accuracy("sample_investment_news.csv")
        elif sys.argv[1] == "--single":
            # Run single configuration with accuracy scores
            model = sys.argv[2] if len(sys.argv) > 2 else "en_core_web_trf"
            use_rule = sys.argv[3].lower() == "true" if len(sys.argv) > 3 else True
            run_complete_pipeline_with_accuracy("sample_investment_news.csv", model, use_rule)
        else:
            print("Usage:")
            print("  python pipeinvest.py --compare                    # Compare all models with accuracy scores")
            print("  python pipeinvest.py --single [model] [true/false] # Run single config with accuracy scores")
            print("  python pipeinvest.py                             # Run default config with accuracy scores")
            print("\nAvailable models: en_core_web_trf (best), en_core_web_md, en_core_web_sm")
    else:
        # Default: run with transformer model and new rule, including accuracy assessment
        print("Running default configuration with transformer model and accuracy scores...")
        run_complete_pipeline_with_accuracy("sample_investment_news.csv")