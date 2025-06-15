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
}

interface JobPrediction extends JobData {
  fraud_probability: number;
  is_fraud: boolean;
  fraud_indicators: string[];
  risk_level: 'High' | 'Medium' | 'Low';
}

export function SuspiciousJobs({ jobs }: { jobs: JobPrediction[] }) {
  const formatProbability = (probability: number): string => {
    // Ensure probability is within valid range 0-100
    const clampedProbability = Math.max(0, Math.min(probability, 100));
    return clampedProbability.toFixed(1);
  };

  const getProbabilityColor = (probability: number): string => {
    if (probability >= 70) return 'text-red-600';
    if (probability >= 30) return 'text-yellow-600';
    return 'text-green-600';
  };

  return (
    <div className="space-y-4">
      {jobs.map((job, index) => (
        <div key={index} className="bg-red-50 rounded-lg p-4 border border-red-100">
          <div className="flex justify-between items-start">
            <div className="flex-1">
              <h4 className="font-semibold text-gray-900 line-clamp-1">{job.title}</h4>
              <p className="text-sm text-gray-500">{job.location}</p>
              {job.salary_range && (
                <p className="text-sm text-gray-600 mt-1">{job.salary_range}</p>
              )}
            </div>
            <div className="flex flex-col items-end ml-4">
              <span className={`px-2 py-1 rounded text-sm font-medium ${
                job.risk_level === 'High' ? 'bg-red-200 text-red-800' :
                job.risk_level === 'Medium' ? 'bg-yellow-200 text-yellow-800' :
                'bg-green-200 text-green-800'
              }`}>
                {job.risk_level} Risk
              </span>
              <span className={`font-medium mt-2 ${getProbabilityColor(job.fraud_probability)}`}>
                {formatProbability(job.fraud_probability)}% fraud risk
              </span>
            </div>
          </div>
          
          {job.description && (
            <div className="mt-3 pt-3 border-t border-red-100">
              <p className="text-sm text-gray-700 line-clamp-2">{job.description}</p>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}