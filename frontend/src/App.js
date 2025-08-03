import React, { useState, useEffect } from "react";
import "./App.css";
import axios from "axios";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

function App() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [dragOver, setDragOver] = useState(false);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    validateAndSetFile(selectedFile);
  };

  const validateAndSetFile = (selectedFile) => {
    if (selectedFile) {
      if (selectedFile.type !== 'text/csv' && !selectedFile.name.toLowerCase().endsWith('.csv')) {
        setError('Please select a CSV file');
        return;
      }
      if (selectedFile.size > 50 * 1024 * 1024) { // 50MB limit
        setError('File size must be less than 50MB');
        return;
      }
      setFile(selectedFile);
      setError(null);
      setResults(null);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const droppedFile = e.dataTransfer.files[0];
    validateAndSetFile(droppedFile);
  };

  const uploadFile = async () => {
    if (!file) {
      setError('Please select a file first');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${API}/upload-csv`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setResults(response.data);
    } catch (err) {
      console.error('Upload error:', err);
      setError(err.response?.data?.detail || 'Error uploading file');
    } finally {
      setLoading(false);
    }
  };

  const resetUpload = () => {
    setFile(null);
    setResults(null);
    setError(null);
  };

  const StatCard = ({ title, value, subtitle }) => (
    <div className="bg-white rounded-lg shadow-md p-6 border-l-4 border-blue-500">
      <h3 className="text-lg font-semibold text-gray-700 mb-2">{title}</h3>
      <p className="text-3xl font-bold text-blue-600 mb-1">{value}</p>
      {subtitle && <p className="text-sm text-gray-500">{subtitle}</p>}
    </div>
  );

  const TableCard = ({ title, data, maxRows = 10 }) => (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h3 className="text-lg font-semibold text-gray-700 mb-4">{title}</h3>
      <div className="overflow-x-auto">
        <table className="min-w-full table-auto">
          <thead>
            <tr className="bg-gray-50">
              <th className="px-4 py-2 text-left text-sm font-medium text-gray-500">Column</th>
              <th className="px-4 py-2 text-left text-sm font-medium text-gray-500">Value</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(data).slice(0, maxRows).map(([key, value], index) => (
              <tr key={key} className={index % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                <td className="px-4 py-2 text-sm font-medium text-gray-900">{key}</td>
                <td className="px-4 py-2 text-sm text-gray-700">
                  {typeof value === 'object' ? JSON.stringify(value, null, 2) : String(value)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <h1 className="text-3xl font-bold text-gray-900">Auto EDA</h1>
              </div>
              <div className="ml-4">
                <p className="text-sm text-gray-500">Automated Exploratory Data Analysis</p>
              </div>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {!results ? (
          /* Upload Section */
          <div className="max-w-3xl mx-auto">
            <div className="bg-white rounded-lg shadow-lg p-8">
              <h2 className="text-2xl font-bold text-gray-900 mb-6 text-center">
                Upload Your CSV Dataset
              </h2>
              
              {/* File Upload Area */}
              <div
                className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
                  dragOver
                    ? 'border-blue-400 bg-blue-50'
                    : 'border-gray-300 hover:border-gray-400'
                }`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
              >
                <div className="space-y-4">
                  <div className="flex justify-center">
                    <svg className="w-16 h-16 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                  </div>
                  
                  {file ? (
                    <div className="space-y-2">
                      <p className="text-lg font-medium text-green-600">
                        ✓ {file.name}
                      </p>
                      <p className="text-sm text-gray-500">
                        {(file.size / 1024 / 1024).toFixed(2)} MB
                      </p>
                    </div>
                  ) : (
                    <div className="space-y-2">
                      <p className="text-lg text-gray-600">
                        Drag and drop your CSV file here, or click to select
                      </p>
                      <p className="text-sm text-gray-500">
                        Supports CSV files up to 50MB
                      </p>
                    </div>
                  )}
                  
                  <input
                    type="file"
                    accept=".csv"
                    onChange={handleFileChange}
                    className="hidden"
                    id="file-upload"
                  />
                  <label
                    htmlFor="file-upload"
                    className="inline-flex items-center px-6 py-3 bg-blue-600 text-white font-medium rounded-md hover:bg-blue-700 cursor-pointer transition-colors"
                  >
                    Select CSV File
                  </label>
                </div>
              </div>

              {/* Error Message */}
              {error && (
                <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-md">
                  <p className="text-red-700">{error}</p>
                </div>
              )}

              {/* Upload Button */}
              {file && (
                <div className="mt-6 flex justify-center space-x-4">
                  <button
                    onClick={uploadFile}
                    disabled={loading}
                    className="px-8 py-3 bg-green-600 text-white font-medium rounded-md hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    {loading ? (
                      <div className="flex items-center">
                        <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        Analyzing...
                      </div>
                    ) : (
                      'Start Analysis'
                    )}
                  </button>
                  <button
                    onClick={resetUpload}
                    className="px-8 py-3 bg-gray-600 text-white font-medium rounded-md hover:bg-gray-700 transition-colors"
                  >
                    Reset
                  </button>
                </div>
              )}
            </div>
          </div>
        ) : (
          /* Results Section */
          <div className="space-y-8">
            {/* Results Header */}
            <div className="bg-white rounded-lg shadow-md p-6">
              <div className="flex justify-between items-center">
                <div>
                  <h2 className="text-2xl font-bold text-gray-900">Analysis Results</h2>
                  <p className="text-gray-600 mt-1">Dataset: {results.filename}</p>
                </div>
                <button
                  onClick={resetUpload}
                  className="px-6 py-2 bg-blue-600 text-white font-medium rounded-md hover:bg-blue-700 transition-colors"
                >
                  Upload New File
                </button>
              </div>
            </div>

            {/* Basic Statistics */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <StatCard
                title="Dataset Shape"
                value={`${results.analysis.basic_info.shape[0]} × ${results.analysis.basic_info.shape[1]}`}
                subtitle="Rows × Columns"
              />
              <StatCard
                title="Total Columns"
                value={results.analysis.basic_info.shape[1]}
                subtitle={`${Object.keys(results.analysis.summary_statistics || {}).length} numerical`}
              />
              <StatCard
                title="Memory Usage"
                value={`${(results.analysis.basic_info.memory_usage / 1024 / 1024).toFixed(2)} MB`}
                subtitle="In Memory"
              />
              <StatCard
                title="Missing Values"
                value={Object.values(results.analysis.basic_info.null_counts || {}).reduce((a, b) => a + b, 0)}
                subtitle="Total null values"
              />
            </div>

            {/* Data Types and Missing Values */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <TableCard
                title="Data Types"
                data={results.analysis.basic_info.dtypes || {}}
              />
              <TableCard
                title="Missing Values (%)"
                data={results.analysis.basic_info.null_percentages || {}}
              />
            </div>

            {/* Visualizations */}
            {results.visualizations && Object.keys(results.visualizations).length > 0 && (
              <div className="space-y-6">
                <h3 className="text-xl font-bold text-gray-900">Visualizations</h3>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {Object.entries(results.visualizations).map(([key, imageData]) => (
                    <div key={key} className="bg-white rounded-lg shadow-md p-6">
                      <h4 className="text-lg font-semibold text-gray-700 mb-4 capitalize">
                        {key.replace(/_/g, ' ')}
                      </h4>
                      <img
                        src={imageData}
                        alt={key}
                        className="w-full h-auto rounded-lg border"
                      />
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Summary Statistics */}
            {results.analysis.summary_statistics && Object.keys(results.analysis.summary_statistics).length > 0 && (
              <div className="bg-white rounded-lg shadow-md p-6">
                <h3 className="text-lg font-semibold text-gray-700 mb-4">Summary Statistics</h3>
                <div className="overflow-x-auto">
                  <table className="min-w-full table-auto">
                    <thead>
                      <tr className="bg-gray-50">
                        <th className="px-4 py-2 text-left text-sm font-medium text-gray-500">Statistic</th>
                        {Object.keys(results.analysis.summary_statistics).map(col => (
                          <th key={col} className="px-4 py-2 text-left text-sm font-medium text-gray-500">{col}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'].map((stat, index) => (
                        <tr key={stat} className={index % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                          <td className="px-4 py-2 text-sm font-medium text-gray-900">{stat}</td>
                          {Object.keys(results.analysis.summary_statistics).map(col => (
                            <td key={col} className="px-4 py-2 text-sm text-gray-700">
                              {results.analysis.summary_statistics[col][stat]?.toFixed?.(2) || 
                               results.analysis.summary_statistics[col][stat] || '-'}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Outliers */}
            {results.analysis.outlier_detection && Object.keys(results.analysis.outlier_detection).length > 0 && (
              <div className="bg-white rounded-lg shadow-md p-6">
                <h3 className="text-lg font-semibold text-gray-700 mb-4">Outlier Detection</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {Object.entries(results.analysis.outlier_detection).map(([col, data]) => (
                    <div key={col} className="border rounded-lg p-4">
                      <h4 className="font-medium text-gray-900 mb-2">{col}</h4>
                      <div className="space-y-1 text-sm text-gray-600">
                        <p>Outliers: {data.count} ({data.percentage.toFixed(1)}%)</p>
                        <p>Range: [{data.lower_bound.toFixed(2)}, {data.upper_bound.toFixed(2)}]</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

export default App;