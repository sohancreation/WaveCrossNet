import numpy as np
from scipy import stats

def compute_statistical_significance(metrics_dict_across_seeds, baseline_name='ResNet1D', target_name='WaveCrossNet'):
    """
    Computes parametric (Paired Student's t-test) and non-parametric (Wilcoxon signed-rank test)
    hypothesis testing metrics comparing target architecture against baselines across random seeds.
    """
    results = {}
    
    target_f1s = np.array(metrics_dict_across_seeds[target_name]['f1_scores'])
    target_mean = np.mean(target_f1s) * 100.0
    target_std = np.std(target_f1s) * 100.0
    
    results[target_name] = {
        'mean_f1': target_mean,
        'std_f1': target_std,
        'formatted': f"{target_mean:.2f}% ± {target_std:.2f}%"
    }
    
    for model_name, data in metrics_dict_across_seeds.items():
        if model_name == target_name:
            continue
            
        base_f1s = np.array(data['f1_scores'])
        base_mean = np.mean(base_f1s) * 100.0
        base_std = np.std(base_f1s) * 100.0
        
        # Paired t-test
        t_stat, p_val_t = stats.ttest_rel(target_f1s, base_f1s)
        
        # Wilcoxon signed-rank test
        try:
            w_stat, p_val_w = stats.wilcoxon(target_f1s, base_f1s)
        except Exception:
            w_stat, p_val_w = 0.0, 1.0
            
        results[model_name] = {
            'mean_f1': base_mean,
            'std_f1': base_std,
            'formatted': f"{base_mean:.2f}% ± {base_std:.2f}%",
            't_stat': float(t_stat),
            'p_val_t': float(p_val_t),
            'w_stat': float(w_stat),
            'p_val_w': float(p_val_w),
            'significant': bool(p_val_t < 0.05)
        }
        
    return results
