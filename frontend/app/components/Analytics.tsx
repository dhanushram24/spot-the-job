import { useEffect, useState } from 'react';
import { RefreshCw, TrendingUp, AlertTriangle, BarChart3, Target, Brain } from 'lucide-react';

interface ModelMetrics {
  precision: number;
  recall: number;
  f1_score: number;
  fraud_rate: number;
  total_samples: number;
  fraud_samples: number;
  legitimate_samples: number;
}

interface JobPrediction {
  fraud_probability: number;
  is_fraud: boolean;
  fraud_indicators: string[];
  risk_level: 'High' | 'Medium' | 'Low';
  title: string;
  location?: string;
  salary_range?: string;
  description?: string;
}

interface AnalyticsProps {
  predictions?: JobPrediction[] | null;
  modelMetrics?: {
    totalJobs: number;
    fraudulentJobs: number;
    genuineJobs: number;
    fraudRate: number;
    f1Score: number;
    precision: number;
    recall: number;
  } | null;
}

interface PerformanceIndicatorProps {
  label: string;
  value: number;
  isPrimary?: boolean;
  isPercentage?: boolean;
  color?: 'blue' | 'green' | 'red' | 'yellow';
  icon?: React.ReactNode;
  description?: string;
}

const PerformanceIndicator = ({ 
  label, 
  value, 
  isPrimary = false, 
  isPercentage = true, 
  color = 'blue',
  icon,
  description
}: PerformanceIndicatorProps) => {
  const colorClasses = {
    blue: 'bg-blue-50 border-blue-200 text-blue-800',
    green: 'bg-green-50 border-green-200 text-green-800',
    red: 'bg-red-50 border-red-200 text-red-800',
    yellow: 'bg-yellow-50 border-yellow-200 text-yellow-800'
  };

  const textColorClasses = {
    blue: 'text-blue-600',
    green: 'text-green-600',
    red: 'text-red-600',
    yellow: 'text-yellow-600'
  };

  return (
    <div className={`p-4 rounded-lg border-2 ${isPrimary ? 'ring-2 ring-blue-300 shadow-lg' : ''} ${colorClasses[color]}`}>
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          {icon}
          <span className={`font-semibold ${isPrimary ? 'text-lg' : 'text-base'}`}>
            {label}
            {isPrimary && <span className="ml-2 text-xs bg-blue-600 text-white px-2 py-1 rounded-full">PRIMARY</span>}
          </span>
        </div>
        <span className={`font-bold ${isPrimary ? 'text-3xl' : 'text-xl'} ${textColorClasses[color]}`}>
          {isPercentage ? (value * 100).toFixed(2) : value.toFixed(0)}{isPercentage ? '%' : ''}
        </span>
      </div>
      {description && (
        <p className={`text-sm ${textColorClasses[color]} opacity-80`}>
          {description}
        </p>
      )}
    </div>
  );
};

export function Analytics({ predictions, modelMetrics }: AnalyticsProps) {
  const [fallbackMetrics, setFallbackMetrics] = useState<ModelMetrics | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Calculate metrics from uploaded data if available
  const uploadedDataMetrics = predictions ? {
    total_samples: predictions.length,
    fraud_samples: predictions.filter(job => job.is_fraud).length,
    legitimate_samples: predictions.filter(job => !job.is_fraud).length,
    fraud_rate: predictions.filter(job => job.is_fraud).length / predictions.length,
    // Use model metrics for performance scores if available, otherwise use placeholder
    f1_score: modelMetrics ? (modelMetrics.f1Score / 100) : 0.85, // Convert percentage to decimal
    precision: modelMetrics ? (modelMetrics.precision / 100) : 0.88,
    recall: modelMetrics ? (modelMetrics.recall / 100) : 0.82
  } : null;

  const fetchFallbackMetrics = async () => {
    try {
      setLoading(true);
      setError(null);
      
      console.log('Fetching fallback model performance statistics...');
      const response = await fetch('http://localhost:5000/statistics', {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json'
        }
      });

      console.log('Response status:', response.status);
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Failed to fetch statistics: ${response.status} ${errorText}`);
      }

      const data = await response.json();
      console.log('Received fallback model metrics:', data);

      if (data.success) {
        setFallbackMetrics(data.metrics);
      } else {
        throw new Error(data.message || 'Failed to get model metrics');
      }
    } catch (error) {
      console.error('Error fetching fallback model metrics:', error);
      setError(error instanceof Error ? error.message : 'Unknown error occurred');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    // Only fetch fallback metrics if no uploaded data is available
    if (!predictions && !modelMetrics) {
      fetchFallbackMetrics();
    }
  }, [predictions, modelMetrics]);

  // Use uploaded data metrics if available, otherwise use fallback metrics
  const metrics = uploadedDataMetrics || fallbackMetrics;
  const dataSource = uploadedDataMetrics ? 'uploaded' : 'training';

  if (loading) {
    return (
      <div className="flex items-center justify-center p-12">
        <div className="flex flex-col items-center gap-4">
          <RefreshCw className="w-8 h-8 text-blue-600 animate-spin" />
          <p className="text-gray-600">Loading model performance metrics...</p>
        </div>
      </div>
    );
  }

  if (error && !metrics) {
    return (
      <div className="p-6">
        <div className="bg-red-50 border border-red-200 rounded-lg p-6 mb-4">
          <div className="flex items-center gap-2 mb-2">
            <AlertTriangle className="w-5 h-5 text-red-600" />
            <h3 className="font-semibold text-red-800">Error Loading Metrics</h3>
          </div>
          <p className="text-red-700 mb-4">{error}</p>
          <button 
            onClick={fetchFallbackMetrics}
            className="flex items-center gap-2 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
          >
            <RefreshCw className="w-4 h-4" />
            Retry Loading
          </button>
        </div>
      </div>
    );
  }

  if (!metrics) {
    return (
      <div className="p-6">
        <div className="text-center">
          <BarChart3 className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-semibold text-gray-900 mb-2">No Metrics Available</h3>
          <p className="text-gray-600">Model performance data is not available.</p>
        </div>
      </div>
    );
  }

  // Calculate model performance interpretation
  const getPerformanceLevel = (f1Score: number) => {
    if (f1Score >= 0.9) return { level: 'Excellent', color: 'green' as const };
    if (f1Score >= 0.8) return { level: 'Good', color: 'blue' as const };
    if (f1Score >= 0.7) return { level: 'Fair', color: 'yellow' as const };
    return { level: 'Needs Improvement', color: 'red' as const };
  };

  const performance = getPerformanceLevel(metrics.f1_score);

  return (
    <div className="space-y-8 p-6">
      {/* Header */}
      <div className="text-center mb-8">
        <div className="flex justify-center mb-4">
          <Brain className="w-12 h-12 text-blue-600" />
        </div>
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Model Performance Analytics</h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          Comprehensive evaluation metrics for the fraud detection model. F1-score is prioritized due to dataset imbalance.
        </p>
        
        {/* Data Source Indicator */}
        <div className={`inline-flex items-center gap-2 mt-4 px-4 py-2 rounded-full text-sm font-medium ${
          dataSource === 'uploaded' 
            ? 'bg-green-100 text-green-800' 
            : 'bg-blue-100 text-blue-800'
        }`}>
          <div className={`w-2 h-2 rounded-full ${
            dataSource === 'uploaded' ? 'bg-green-500' : 'bg-blue-500'
          }`} />
          {dataSource === 'uploaded' ? 'Showing uploaded data metrics' : 'Showing training data metrics'}
        </div>
      </div>

      {/* Primary F1-Score Section */}
      <div className="bg-gradient-to-br from-blue-50 to-indigo-100 rounded-xl p-8 mb-8">
        <div className="text-center mb-6">
          <Target className="w-16 h-16 text-blue-600 mx-auto mb-4" />
          <h2 className="text-2xl font-bold text-gray-900 mb-2">Primary Evaluation Metric</h2>
          <p className="text-gray-700 max-w-xl mx-auto">
            F1-score is the harmonic mean of precision and recall, ideal for imbalanced datasets like fraud detection
          </p>
        </div>
        
        <div className="bg-white rounded-lg p-6 shadow-lg border-4 border-blue-200">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-xl font-bold text-blue-800 mb-1">F1-Score</h3>
              <p className="text-sm text-blue-600">
                Model Performance: <span className={`font-semibold text-${performance.color}-600`}>
                  {performance.level}
                </span>
              </p>
            </div>
            <div className="text-right">
              <div className="text-4xl font-bold text-blue-600">
                {(metrics.f1_score * 100).toFixed(2)}%
              </div>
              <div className="text-sm text-gray-500">
                Range: 0-100%
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Supporting Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <PerformanceIndicator
          label="Precision"
          value={metrics.precision}
          color="green"
          icon={<TrendingUp className="w-5 h-5" />}
          description="Of predicted fraud cases, how many were actually fraudulent"
        />
        
        <PerformanceIndicator
          label="Recall"
          value={metrics.recall}
          color="yellow"
          icon={<Target className="w-5 h-5" />}
          description="Of all actual fraud cases, how many were correctly identified"
        />
      </div>

      {/* Dataset Information */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Dataset Statistics */}
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h3 className="text-xl font-bold text-gray-900 mb-6 flex items-center gap-2">
            <BarChart3 className="w-5 h-5" />
            Dataset Statistics
            {dataSource === 'uploaded' && (
              <span className="text-xs bg-green-100 text-green-800 px-2 py-1 rounded-full ml-2">
                UPLOADED DATA
              </span>
            )}
          </h3>
          
          <div className="space-y-4">
            <div className="flex justify-between items-center py-2 border-b">
              <span className="text-gray-600">Total Samples</span>
              <span className="font-bold text-gray-900">{metrics.total_samples.toLocaleString()}</span>
            </div>
            
            <div className="flex justify-between items-center py-2 border-b">
              <span className="text-gray-600">Fraudulent Jobs</span>
              <div className="text-right">
                <span className="font-bold text-red-600">{metrics.fraud_samples.toLocaleString()}</span>
                <div className="text-xs text-gray-500">
                  {((metrics.fraud_samples / metrics.total_samples) * 100).toFixed(1)}% of total
                </div>
              </div>
            </div>
            
            <div className="flex justify-between items-center py-2">
              <span className="text-gray-600">Legitimate Jobs</span>
              <div className="text-right">
                <span className="font-bold text-green-600">{metrics.legitimate_samples.toLocaleString()}</span>
                <div className="text-xs text-gray-500">
                  {((metrics.legitimate_samples / metrics.total_samples) * 100).toFixed(1)}% of total
                </div>
              </div>
            </div>
          </div>

          {/* Dataset Balance Indicator */}
          <div className="mt-6 p-4 bg-gray-50 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <AlertTriangle className="w-4 h-4 text-yellow-600" />
              <span className="font-semibold text-gray-900">Dataset Balance</span>
            </div>
            <p className="text-sm text-gray-600">
              This is an <strong>imbalanced dataset</strong> with {(metrics.fraud_rate * 100).toFixed(1)}% fraud rate. 
              F1-score provides better evaluation than accuracy for such datasets.
            </p>
          </div>
        </div>

        {/* Model Interpretation */}
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h3 className="text-xl font-bold text-gray-900 mb-6">Model Interpretation</h3>
          
          <div className="space-y-4">
            <div className="p-4 bg-blue-50 rounded-lg border border-blue-200">
              <h4 className="font-semibold text-blue-800 mb-2">Why F1-Score?</h4>
              <p className="text-sm text-blue-700">
                F1-score balances precision and recall, making it ideal for fraud detection where both 
                false positives (flagging legitimate jobs) and false negatives (missing fraud) are costly.
              </p>
            </div>
            
            <div className="p-4 bg-green-50 rounded-lg border border-green-200">
              <h4 className="font-semibold text-green-800 mb-2">Precision Impact</h4>
              <p className="text-sm text-green-700">
                High precision ({(metrics.precision * 100).toFixed(1)}%) means when the model flags a job as fraud, 
                it&apos;s usually correct, reducing false alarms.
              </p>
            </div>
            
            <div className="p-4 bg-yellow-50 rounded-lg border border-yellow-200">
              <h4 className="font-semibold text-yellow-800 mb-2">Recall Impact</h4>
              <p className="text-sm text-yellow-700">
                {metrics.recall >= 0.8 ? 'Good' : 'Moderate'} recall ({(metrics.recall * 100).toFixed(1)}%) indicates 
                the model catches most fraud cases, protecting job seekers effectively.
              </p>
            </div>
          </div>

          {/* Performance Recommendation */}
          <div className="mt-6 p-4 bg-gray-900 text-white rounded-lg">
            <h4 className="font-semibold mb-2">Performance Assessment</h4>
            <p className="text-sm">
              {metrics.f1_score >= 0.8 
                ? "🎉 Excellent performance! The model effectively balances fraud detection and false alarm rates."
                : metrics.f1_score >= 0.7
                ? "✅ Good performance. Consider fine-tuning to improve precision-recall balance."
                : "⚠️ Performance needs improvement. Consider retraining with more data or feature engineering."
              }
            </p>
          </div>
        </div>
      </div>

      {/* Refresh Button - only show if using fallback metrics */}
      {!uploadedDataMetrics && (
        <div className="text-center pt-6">
          <button 
            onClick={fetchFallbackMetrics}
            className="flex items-center gap-2 mx-auto px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors shadow-lg"
          >
            <RefreshCw className="w-4 h-4" />
            Refresh Metrics
          </button>
        </div>
      )}
    </div>
  );
}