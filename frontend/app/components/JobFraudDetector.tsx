"use client";

import { useState, useMemo } from 'react';
import { Shield, Upload, BarChart3, TrendingUp, FileText, RefreshCw, 
         AlertTriangle, CheckCircle, Download, XCircle } from 'lucide-react';
import { LucideIcon } from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  PieChart as RechartsPieChart,
  Pie,
  Cell
} from 'recharts';
import { Analytics } from './Analytics';

interface JobData {
  job_id?: number;
  title: string;
  description?: string;
  salary_range?: string;
  required_experience?: string;
  required_education?: string;
  company_profile?: string;
  telecommuting?: boolean;
  has_company_logo?: boolean;
  location?: string;
  company?: string;
  employment_type?: string;
}

interface JobPrediction extends JobData {
  fraud_probability: number;
  is_fraud: boolean;
  fraud_indicators?: string[];
  risk_level: 'High' | 'Medium' | 'Low';
}

interface ModelMetrics {
  totalJobs: number;
  fraudulentJobs: number;
  genuineJobs: number;
  fraudRate?: number;
  f1Score?: number;
  precision?: number;
  recall?: number;
}

const JobFraudDetector = () => {
  const [uploadedData, setUploadedData] = useState<JobData[] | null>(null);
  const [predictions, setPredictions] = useState<JobPrediction[] | null>(null);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [activeTab, setActiveTab] = useState<'upload' | 'results' | 'analytics'>('upload');
  const [selectedJob, setSelectedJob] = useState<JobPrediction | null>(null);
  const [modelMetrics, setModelMetrics] = useState<ModelMetrics | null>(null);
  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setIsProcessing(true);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Prediction failed');
      }

      const result = await response.json();
      
      if (!result.success) {
        throw new Error(result.message || 'Processing failed');
      }
      
      setUploadedData(result.predictions);
      setPredictions(result.predictions);
      setModelMetrics(result.summary);
      setActiveTab('results');
    } catch (error: unknown) {
      console.error('Error:', error);
      if (error instanceof Error) {
        alert(error.message || 'Failed to process file');
      } else {
        alert('Failed to process file');
      }
    } finally {
      setIsProcessing(false);
    }
  };

  const fraudProbabilityData = useMemo(() => {
    if (!predictions) return [];
    const bins = Array(10).fill(0);
    predictions.forEach(job => {
      const bin = Math.min(9, Math.floor(job.fraud_probability * 10));
      bins[bin]++;
    });
    return bins.map((count, index) => ({
      range: `${index * 10}-${(index + 1) * 10}%`,
      count
    }));
  }, [predictions]);

  const pieChartData = useMemo(() => {
    if (!predictions) return [];
    const fraudulent = predictions.filter(job => job.is_fraud).length;
    const genuine = predictions.length - fraudulent;
    return [
      { name: 'Genuine', value: genuine, color: '#22c55e' },
      { name: 'Fraudulent', value: fraudulent, color: '#ef4444' }
    ];
  }, [predictions]);

  const topSuspiciousJobs = useMemo(() => {
    if (!predictions) return [];
    return [...predictions]
      .sort((a, b) => b.fraud_probability - a.fraud_probability)
      .slice(0, 10);
  }, [predictions]);

  interface TabButtonProps {
    id: string;
    icon: LucideIcon;
    label: string;
    isActive: boolean;
    onClick: () => void;
  }

  const TabButton = ({ icon: Icon, label, isActive, onClick }: TabButtonProps) => (
    <button
      onClick={onClick}
      className={`flex items-center gap-2 px-6 py-3 rounded-lg font-medium transition-all ${
        isActive 
          ? 'bg-blue-600 text-white shadow-lg' 
          : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
      }`}
    >
      <Icon size={20} />
      {label}
    </button>
  );

  interface JobCardProps {
    job: JobPrediction;
    onClick: (job: JobPrediction) => void;
  }

  const JobCard = ({ job, onClick }: JobCardProps) => (
    <div 
      onClick={() => onClick(job)}
      className={`p-4 border rounded-lg cursor-pointer transition-all hover:shadow-md ${
        job.risk_level === 'High' ? 'border-red-300 bg-red-50' :
        job.risk_level === 'Medium' ? 'border-yellow-300 bg-yellow-50' :
        'border-green-300 bg-green-50'
      }`}
    >
      <div className="flex justify-between items-start mb-2">
        <h3 className="font-semibold text-gray-900 line-clamp-1">{job.title}</h3>
        <div className={`px-2 py-1 rounded text-xs font-medium ${
          job.risk_level === 'High' ? 'bg-red-200 text-red-800' :
          job.risk_level === 'Medium' ? 'bg-yellow-200 text-yellow-800' :
          'bg-green-200 text-green-800'
        }`}>
          {job.risk_level} Risk
        </div>
      </div>
      
      {/* Company and Location */}
      <div className="space-y-1 mb-3">
        {job.company && (
          <p className="text-sm text-gray-600 font-medium">{job.company}</p>
        )}
        <p className="text-sm text-gray-500">{job.location}</p>
      </div>

      {/* Salary Range - Now properly displayed */}
      {job.salary_range && job.salary_range !== 'Not specified' && (
        <div className="flex items-center gap-1 mb-2">
          <span className="text-sm font-medium text-green-700">{job.salary_range}</span>
        </div>
      )}

      <div className="flex justify-between items-center">
        <span className="text-sm font-medium text-gray-700">
          {job.fraud_probability.toFixed(1)}% Fraud Risk
        </span>
        {job.is_fraud ? (
          <XCircle className="text-red-500" size={16} />
        ) : (
          <CheckCircle className="text-green-500" size={16} />
        )}
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex justify-center mb-4">
            <Shield className="text-blue-600" size={48} />
          </div>
          <h1 className="text-4xl font-bold text-gray-900 mb-2">Job Fraud Detection System</h1>
          <p className="text-gray-600 max-w-2xl mx-auto">
            Protect job seekers from fraudulent listings using advanced machine learning algorithms. 
            Upload your job data and get instant fraud predictions with detailed insights.
          </p>
        </div>

        {/* Navigation Tabs */}
        <div className="flex justify-center gap-4 mb-8">
          <TabButton 
            id="upload" 
            icon={Upload} 
            label="Upload Data" 
            isActive={activeTab === 'upload'} 
            onClick={() => setActiveTab('upload')} 
          />
          <TabButton 
            id="results" 
            icon={BarChart3} 
            label="Results Dashboard" 
            isActive={activeTab === 'results'} 
            onClick={() => setActiveTab('results')} 
          />
          <TabButton 
            id="analytics" 
            icon={TrendingUp} 
            label="Analytics" 
            isActive={activeTab === 'analytics'} 
            onClick={() => setActiveTab('analytics')} 
          />
        </div>

        {/* Upload Tab */}
        {activeTab === 'upload' && (
          <div className="max-w-2xl mx-auto">
            <div className="bg-white rounded-2xl shadow-xl p-8">
              <div className="text-center mb-6">
                <FileText className="mx-auto text-blue-600 mb-4" size={48} />
                <h2 className="text-2xl font-bold text-gray-900 mb-2">Upload Job Data</h2>
                <p className="text-gray-600">Upload a CSV file containing job listings to detect potential fraud</p>
              </div>

              <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-blue-400 transition-colors">
                <input
                  type="file"
                  accept=".csv"
                  onChange={handleFileUpload}
                  className="hidden"
                  id="file-upload"
                  disabled={isProcessing}
                />
                <label htmlFor="file-upload" className={`cursor-pointer ${isProcessing ? 'opacity-50' : ''}`}>
                  {isProcessing ? (
                    <RefreshCw className="mx-auto text-blue-600 mb-4 animate-spin" size={48} />
                  ) : (
                    <Upload className="mx-auto text-gray-400 mb-4" size={48} />
                  )}
                  <p className="text-lg font-medium text-gray-900 mb-2">
                    {isProcessing ? 'Processing...' : 'Choose CSV file'}
                  </p>
                  <p className="text-sm text-gray-500">
                    {isProcessing ? 'Analyzing job listings for fraud patterns' : 'Upload your job listings CSV file'}
                  </p>
                </label>
              </div>

              {uploadedData && (
                <div className="mt-6 p-4 bg-green-50 border border-green-200 rounded-lg">
                  <p className="text-green-800 font-medium">
                    ✅ Successfully uploaded {uploadedData.length} job listings
                  </p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Results Dashboard Tab */}
        {activeTab === 'results' && predictions && (
          <div className="space-y-8">
            {/* Overview Cards */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
              <div className="bg-white rounded-xl shadow-lg p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600">Total Jobs</p>
                    <p className="text-3xl font-bold text-gray-900">{modelMetrics?.totalJobs}</p>
                  </div>
                  <FileText className="text-blue-600" size={32} />
                </div>
              </div>
              
              <div className="bg-white rounded-xl shadow-lg p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600">Fraudulent</p>
                    <p className="text-3xl font-bold text-red-600">{modelMetrics?.fraudulentJobs}</p>
                  </div>
                  <AlertTriangle className="text-red-600" size={32} />
                </div>
              </div>
              
              <div className="bg-white rounded-xl shadow-lg p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600">Genuine</p>
                    <p className="text-3xl font-bold text-green-600">{modelMetrics?.genuineJobs}</p>
                  </div>
                  <CheckCircle className="text-green-600" size={32} />
                </div>
              </div>
              
              <div className="bg-white rounded-xl shadow-lg p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <h1 className="text-sm font-medium text-gray-600">F1 Score</h1>
                   
                  </div>  
                  <TrendingUp className="text-blue-600" size={32} />
                </div>
              </div>
            </div>

            {/* Charts Row */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Fraud Probability Distribution */}
              <div className="bg-white rounded-xl shadow-lg p-6">
                <h3 className="text-xl font-bold text-gray-900 mb-4">Fraud Probability Distribution</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={fraudProbabilityData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="range" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="count" fill="#3b82f6" />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Fraud vs Genuine Pie Chart */}
              <div className="bg-white rounded-xl shadow-lg p-6">
                <h3 className="text-xl font-bold text-gray-900 mb-4">Jobs Classification</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <RechartsPieChart>
                    <Pie
                      data={pieChartData}
                      cx="50%"
                      cy="50%"
                      outerRadius={100}
                      dataKey="value"
                      label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                    >
                      {pieChartData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </RechartsPieChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Top Suspicious Jobs */}
            <div className="bg-white rounded-xl shadow-lg p-6">
              <div className="flex justify-between items-center mb-6">
                <h3 className="text-xl font-bold text-gray-900">Top 10 Most Suspicious Jobs</h3>
                <button className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
                  <Download size={16} />
                  Export Results
                </button>
              </div>
              <div className="grid gap-4">
                {topSuspiciousJobs.map((job, index) => (
                  <JobCard key={index} job={job} onClick={setSelectedJob} />
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Analytics Tab */}
        {activeTab === 'analytics' && (
          <div className="space-y-8">
            <Analytics />
            {predictions && (
              <>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                  {/* Risk Level Distribution */}
                  <div className="bg-white rounded-xl shadow-lg p-6">
                    <h3 className="text-xl font-bold text-gray-900 mb-4">Risk Level Distribution</h3>
                    <div className="space-y-3">
                      {['High', 'Medium', 'Low'].map(level => {
                        const count = predictions.filter(job => job.risk_level === level).length;
                        const percentage = (count / predictions.length * 100).toFixed(1);
                        return (
                          <div key={level} className="flex items-center justify-between">
                            <span className="text-gray-600">{level} Risk</span>
                            <div className="flex items-center gap-2">
                              <span className="text-sm text-gray-500">{count} ({percentage}%)</span>
                              <div className={`w-3 h-3 rounded-full ${
                                level === 'High' ? 'bg-red-500' :
                                level === 'Medium' ? 'bg-yellow-500' : 'bg-green-500'
                              }`} />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>

                {/* Suspicious Jobs Section in Analytics Tab */}
                <div className="bg-white rounded-xl shadow-lg p-6">
                  <h3 className="text-xl font-bold text-gray-900 mb-4">Most Suspicious Jobs</h3>
                  <div className="grid gap-4">
                    {topSuspiciousJobs.slice(0, 5).map((job, index) => (
                      <JobCard key={index} job={job} onClick={setSelectedJob} />
                    ))}
                  </div>
                </div>
              </>
            )}
          </div>
        )}

        {/* Job Detail Modal */}
        {selectedJob && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
            <div className="bg-white rounded-2xl shadow-2xl max-w-2xl w-full max-h-96 overflow-y-auto">
              <div className="p-6">
                <div className="flex justify-between items-start mb-4">
                  <h2 className="text-2xl font-bold text-gray-900">{selectedJob.title}</h2>
                  <button 
                    onClick={() => setSelectedJob(null)}
                    className="text-gray-400 hover:text-gray-600"
                  >
                    ✕
                  </button>
                </div>
                
                <div className="space-y-4">
                  <div>
                    <h3 className="font-semibold text-gray-900">Risk Assessment</h3>
                    <div className={`inline-block px-3 py-1 rounded-full text-sm font-medium ${
                      selectedJob.risk_level === 'High' ? 'bg-red-100 text-red-800' :
                      selectedJob.risk_level === 'Medium' ? 'bg-yellow-100 text-yellow-800' :
                      'bg-green-100 text-green-800'
                    }`}>
                      {selectedJob.risk_level} Risk ({selectedJob.fraud_probability.toFixed(1)}% probability)
                    </div>
                  </div>

                  {selectedJob.company && (
                    <div>
                      <h3 className="font-semibold text-gray-900">Company</h3>
                      <p className="text-gray-600">{selectedJob.company}</p>
                    </div>
                  )}
                  
                  <div>
                    <h3 className="font-semibold text-gray-900">Location</h3>
                    <p className="text-gray-600">{selectedJob.location}</p>
                  </div>
                  
                  {/* Salary Range - Now properly displayed in modal */}
                  {selectedJob.salary_range && selectedJob.salary_range !== 'Not specified' && (
                    <div>
                      <h3 className="font-semibold text-gray-900">Salary Range</h3>
                      <div className="flex items-center gap-2">
                        <p className="text-gray-600 font-medium">{selectedJob.salary_range}</p>
                      </div>
                    </div>
                  )}

                  {selectedJob.employment_type && (
                    <div>
                      <h3 className="font-semibold text-gray-900">Employment Type</h3>
                      <p className="text-gray-600">{selectedJob.employment_type}</p>
                    </div>
                  )}
                  
                  <div>
                    <h3 className="font-semibold text-gray-900">Description</h3>
                    <p className="text-gray-600 text-sm">{selectedJob.description || 'No description available'}</p>
                  </div>
                  
                  {selectedJob.fraud_indicators && selectedJob.fraud_indicators.length > 0 && (
                    <div>
                      <h3 className="font-semibold text-red-600">⚠️ Fraud Indicators</h3>
                      <ul className="list-disc list-inside text-sm text-red-600 space-y-1">
                        {selectedJob.fraud_indicators.map((indicator, index) => (
                          <li key={index}>{indicator}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Show instructions if no data */}
        {!predictions && !isProcessing && activeTab !== 'upload' && (
          <div className="text-center py-12">
            <Shield className="mx-auto text-gray-400 mb-4" size={64} />
            <h3 className="text-xl font-semibold text-gray-900 mb-2">No Data Available</h3>
            <p className="text-gray-600 mb-4">Please upload a CSV file first to see the results</p>
            <button 
              onClick={() => setActiveTab('upload')}
              className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              Go to Upload
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default JobFraudDetector;