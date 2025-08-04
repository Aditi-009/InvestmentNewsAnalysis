from task24 import process_csv as relevance_filter
from bankllm import process_csv_with_llm, add_manual_accuracy_column, load_csv_with_encoding
import pandas as pd
import time
from datetime import datetime

def run_complete_pipeline_with_accuracy(input_csv: str, model_name: str = "en_core_web_trf", 
                                       use_new_rule: bool = True, output_csv: str = None):
    """
    Run complete pipeline with accuracy assessment and updated sentiment analysis
    NOW ONLY OUTPUTS RELEVANT ITEMS
    
    Args:
        input_csv: Input CSV file path
        model_name: spaCy model to use (default: en_core_web_trf for best accuracy)
        use_new_rule: Whether to use new filtering rule
        output_csv: Output file path (auto-generated if None)
    """
    
    if output_csv is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rule_suffix = "new_rule" if use_new_rule else "old_rule"
        output_csv = f"SA_Accuracy_Scores_RELEVANT_ONLY_{model_name}_{rule_suffix}_{timestamp}.csv"
    
    print(f"🚀 Running Complete Pipeline with Numerical Accuracy Scores")
    print(f"📖 Input: {input_csv}")
    print(f"🤖 Model: {model_name}")
    print(f"📏 New Rule: {'Enabled' if use_new_rule else 'Disabled'}")
    print(f"📝 Output: {output_csv}")
    print(f"🎯 Mode: RELEVANT ITEMS ONLY")
    print("-" * 80)
    
    try:
        # Stage 1: Relevance filtering with accuracy scoring
        print("🔍 Stage 1: Filtering relevant news with accuracy scoring...")
        filtered_df = relevance_filter(input_csv, model_name, use_new_rule)
        
        # Store original counts for reporting
        original_total_count = len(filtered_df)
        original_relevant_count = len(filtered_df[filtered_df['relevant'] == 'YES'])
        original_non_relevant_count = len(filtered_df[filtered_df['relevant'] == 'NO'])
        avg_accuracy = filtered_df['accuracy'].mean()
        high_accuracy_count = len(filtered_df[filtered_df['accuracy'] >= 0.7])
        
        print(f"✅ Stage 1 complete:")
        print(f"   Total processed: {original_total_count}")
        print(f"   Relevant: {original_relevant_count}/{original_total_count} ({original_relevant_count/original_total_count*100:.1f}%)")
        print(f"   Non-relevant (will be excluded): {original_non_relevant_count}/{original_total_count} ({original_non_relevant_count/original_total_count*100:.1f}%)")
        print(f"   Average accuracy (rule-based): {avg_accuracy:.3f}")
        print(f"   High accuracy (≥0.7): {high_accuracy_count}/{original_total_count} ({high_accuracy_count/original_total_count*100:.1f}%)")
        
        # Stage 2: Add AI-based accuracy assessment (still on all items before filtering)
        print("\n🤖 Stage 2: Adding AI-based accuracy assessment with scores...")
        result_df = add_manual_accuracy_column(filtered_df)
        
        ai_avg_accuracy = result_df['Accuracy SA'].mean()
        ai_high_accuracy_count = len(result_df[result_df['Accuracy SA'] >= 0.7])
        print(f"✅ Stage 2 complete:")
        print(f"   AI Average Accuracy: {ai_avg_accuracy:.3f}")
        print(f"   AI High Accuracy (≥0.7): {ai_high_accuracy_count}/{original_total_count} ({ai_high_accuracy_count/original_total_count*100:.1f}%)")
        
        # Rename the original accuracy column for clarity
        if 'accuracy' in result_df.columns:
            result_df = result_df.rename(columns={'accuracy': 'accuracy_AP'})
        
        # Stage 3: Sentiment analysis with updated -1 to +1 scale AND filtering to relevant only
        print("\n💭 Stage 3: Analyzing target sentiment (-1 to +1 scale) and filtering to relevant items only...")
        final_df = process_csv_with_llm(result_df)
        
        # Final counts after filtering
        final_count = len(final_df)
        
        print("✅ Stage 3 complete: Target sentiment analysis finished and non-relevant items excluded")
        
        # Stage 4: Save results with UTF-8 encoding
        print(f"\n💾 Stage 4: Saving results to {output_csv}...")
        final_df.to_csv(output_csv, index=False, encoding='utf-8')
        print("✅ Stage 4 complete: Results saved successfully")
        
        # Final summary with updated statistics
        print(f"\n📊 Final Summary:")
        print(f"   Model Used: {model_name}")
        print(f"   New Rule: {'Yes' if use_new_rule else 'No'}")
        print(f"   Original Total Items: {original_total_count}")
        print(f"   Original Relevant Items: {original_relevant_count} ({original_relevant_count/original_total_count*100:.1f}%)")
        print(f"   Items in Final Output: {final_count}")
        print(f"   Excluded Items (non-relevant): {original_total_count - final_count}")
        print(f"   Rule-based Avg Accuracy (original data): {avg_accuracy:.3f}")
        
        # Recalculate accuracy statistics for final output
        if len(final_df) > 0:
            if 'accuracy_AP' in final_df.columns:
                final_rule_accuracy = final_df['accuracy_AP'].mean()
                print(f"   Rule-based Avg Accuracy (final output): {final_rule_accuracy:.3f}")
            
            if 'Accuracy SA' in final_df.columns:
                final_ai_accuracy = final_df['Accuracy SA'].mean()
                print(f"   AI-based Avg Accuracy (final output): {final_ai_accuracy:.3f}")
        
        # Accuracy score distribution for final output
        if len(final_df) > 0:
            if 'accuracy_AP' in final_df.columns:
                rule_accuracy_dist = final_df['accuracy_AP'].describe()
                print(f"   Final Rule-based Accuracy Distribution:")
                print(f"     Min: {rule_accuracy_dist['min']:.3f}, Max: {rule_accuracy_dist['max']:.3f}")
                print(f"     25%: {rule_accuracy_dist['25%']:.3f}, 50%: {rule_accuracy_dist['50%']:.3f}, 75%: {rule_accuracy_dist['75%']:.3f}")
            
            if 'Accuracy SA' in final_df.columns:
                ai_accuracy_dist = final_df['Accuracy SA'].describe()
                print(f"   Final AI-based Accuracy Distribution:")
                print(f"     Min: {ai_accuracy_dist['min']:.3f}, Max: {ai_accuracy_dist['max']:.3f}")
                print(f"     25%: {ai_accuracy_dist['25%']:.3f}, 50%: {ai_accuracy_dist['50%']:.3f}, 75%: {ai_accuracy_dist['75%']:.3f}")
            
            # Target sentiment breakdown (only relevant items)
            if 'target_sentiment' in final_df.columns:
                sentiment_counts = final_df['target_sentiment'].value_counts()
                print(f"   Target Sentiment Distribution (relevant items only):")
                for sentiment, count in sentiment_counts.items():
                    print(f"     {sentiment}: {count} ({count/len(final_df)*100:.1f}%)")
                
                # Show sentiment score statistics
                if 'target_sentiment_score' in final_df.columns:
                    scores = final_df['target_sentiment_score']
                    print(f"   Target Sentiment Score Stats (relevant items only):")
                    print(f"     Mean: {scores.mean():.3f}")
                    print(f"     Range: {scores.min():.3f} to {scores.max():.3f}")
                    print(f"     Std Dev: {scores.std():.3f}")
        
        print(f"\n✅ Complete pipeline finished successfully!")
        print(f"📄 Results saved to: {output_csv}")
        print(f"🎯 Output contains ONLY relevant items ({final_count} items)")
        
        return final_df
        
    except Exception as e:
        print(f"❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_model_comparison_with_accuracy(input_csv: str = "sample_investment_news.csv"):
    """
    Run comprehensive model comparison with accuracy assessment using numerical scores
    NOW ONLY OUTPUTS RELEVANT ITEMS
    """
    
    print("🚀 Starting Comprehensive Model Comparison with Numerical Accuracy Scores")
    print("🎯 Mode: RELEVANT ITEMS ONLY in output files")
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
                    # Note: We need to get original statistics from the unfiltered data
                    # For comparison purposes, we should load and process the original data
                    from task24 import process_csv as relevance_filter
                    original_df = relevance_filter(input_csv, model, use_new_rule)
                    
                    # Calculate comprehensive metrics
                    original_total_items = len(original_df)
                    original_relevant_items = len(original_df[original_df['relevant'] == 'YES'])
                    final_output_items = len(result_df)
                    
                    # Accuracy metrics (from original data for comparison)
                    rule_avg_accuracy = original_df['accuracy'].mean() if 'accuracy' in original_df.columns else 0
                    rule_high_accuracy = len(original_df[original_df['accuracy'] >= 0.7]) if 'accuracy' in original_df.columns else 0
                    
                    # Final output accuracy metrics
                    final_rule_avg_accuracy = result_df['accuracy_AP'].mean() if 'accuracy_AP' in result_df.columns else 0
                    final_ai_avg_accuracy = result_df['Accuracy SA'].mean() if 'Accuracy SA' in result_df.columns else 0
                    final_ai_high_accuracy = len(result_df[result_df['Accuracy SA'] >= 0.7]) if 'Accuracy SA' in result_df.columns else 0
                    
                    # Sentiment statistics (from final output only)
                    sentiment_stats = {}
                    if 'target_sentiment_score' in result_df.columns and len(result_df) > 0:
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
                        'original_total_items': original_total_items,
                        'original_relevant_items': original_relevant_items,
                        'final_output_items': final_output_items,
                        'exclusion_rate': (original_total_items - final_output_items) / original_total_items * 100,
                        'original_rule_avg_accuracy': rule_avg_accuracy,
                        'final_rule_avg_accuracy': final_rule_avg_accuracy,
                        'final_ai_avg_accuracy': final_ai_avg_accuracy,
                        'original_rule_high_accuracy': rule_high_accuracy,
                        'final_ai_high_accuracy': final_ai_high_accuracy,
                        'original_relevance_rate': original_relevant_items / original_total_items * 100,
                        'final_ai_high_accuracy_rate': final_ai_high_accuracy / final_output_items * 100 if final_output_items > 0 else 0,
                        **sentiment_stats
                    }
                    
                    all_results[f"{model}_{rule_name}"] = metrics
                    
                    print(f"✅ Completed {model} with {rule_name}")
                    print(f"   Original Total: {original_total_items}")
                    print(f"   Original Relevant: {original_relevant_items} ({metrics['original_relevance_rate']:.1f}%)")
                    print(f"   Final Output: {final_output_items}")
                    print(f"   Exclusion Rate: {metrics['exclusion_rate']:.1f}%")
                    print(f"   Final AI Avg Accuracy: {final_ai_avg_accuracy:.3f}")
                
            except Exception as e:
                print(f"❌ Failed to test {model} with {rule_name}: {e}")
                continue
    
    # Generate comparison report
    if all_results:
        comparison_df = pd.DataFrame(list(all_results.values()))
        
        # Save comparison report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"comprehensive_accuracy_scores_RELEVANT_ONLY_comparison_{timestamp}.csv"
        comparison_df.to_csv(report_file, index=False)
        
        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE ACCURACY SCORES COMPARISON REPORT")
        print("🎯 RELEVANT ITEMS ONLY OUTPUT MODE")
        print("=" * 80)
        
        print(f"\n📋 Results Summary:")
        display_cols = ['model', 'rule_config', 'original_total_items', 'final_output_items', 'exclusion_rate', 'final_ai_avg_accuracy']
        if all(col in comparison_df.columns for col in display_cols):
            print(comparison_df[display_cols].round(3).to_string(index=False))
        
        # Find best performers
        if len(comparison_df) > 0:
            print(f"\n🏆 Best Performers:")
            if 'final_ai_avg_accuracy' in comparison_df.columns:
                best_ai_accuracy = comparison_df.loc[comparison_df['final_ai_avg_accuracy'].idxmax()]
                print(f"   Best Final AI Avg Accuracy: {best_ai_accuracy['model']} ({best_ai_accuracy['rule_config']}) - {best_ai_accuracy['final_ai_avg_accuracy']:.3f}")
            
            if 'final_output_items' in comparison_df.columns:
                most_relevant = comparison_df.loc[comparison_df['final_output_items'].idxmax()]
                print(f"   Most Relevant Items: {most_relevant['model']} ({most_relevant['rule_config']}) - {most_relevant['final_output_items']} items")
            
            if 'exclusion_rate' in comparison_df.columns:
                least_exclusion = comparison_df.loc[comparison_df['exclusion_rate'].idxmin()]
                print(f"   Least Exclusion: {least_exclusion['model']} ({least_exclusion['rule_config']}) - {least_exclusion['exclusion_rate']:.1f}%")
        
        print(f"\n📄 Detailed comparison saved to: {report_file}")
        print(f"🎯 All output files contain ONLY relevant items")
        return comparison_df
    
    else:
        print("❌ No successful results to compare")
        return None

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--compare":
            # Run full model comparison with accuracy scores (relevant only)
            run_model_comparison_with_accuracy("sample_investment_news.csv")
        elif sys.argv[1] == "--single":
            # Run single configuration with accuracy scores (relevant only)
            model = sys.argv[2] if len(sys.argv) > 2 else "en_core_web_trf"
            use_rule = sys.argv[3].lower() == "true" if len(sys.argv) > 3 else True
            run_complete_pipeline_with_accuracy("sample_investment_news.csv", model, use_rule)
        else:
            print("Usage:")
            print("  python pipeinvest.py --compare                    # Compare all models (relevant items only)")
            print("  python pipeinvest.py --single [model] [true/false] # Run single config (relevant items only)")
            print("  python pipeinvest.py                             # Run default config (relevant items only)")
            print("\nAvailable models: en_core_web_trf (best), en_core_web_md, en_core_web_sm")
            print("\n🎯 NOTE: All output files now contain ONLY relevant items")
    else:
        # Default: run with transformer model and new rule, including accuracy assessment (relevant only)
        print("Running default configuration with transformer model and accuracy scores...")
        print("🎯 Output will contain ONLY relevant items")
        run_complete_pipeline_with_accuracy("sample_investment_news.csv")
