"use client";

import React, { useEffect, useState } from 'react';

interface Issue {
  id: number;
  title: string;
  html_url: string;
  user: {
    login: string;
    avatar_url: string;
    html_url: string;
  };
  state: string;
  created_at: string;
}

const IssuesList: React.FC = () => {
  const [issues, setIssues] = useState<Issue[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    fetch('https://api.github.com/repos/jeacpa/Project-001/issues')
      .then((res) => {
        if (!res.ok) {
          throw new Error('Failed to fetch issues');
        }
        return res.json();
      })
      .then((data) => {
        setIssues(data);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  if (loading) return <p>Loading issues...</p>;
  if (error) return <p>Error: {error}</p>;

  return (
    <div className="p-4">
      <h2 className="text-xl font-bold mb-4">GitHub Issues</h2>
      <ul className="space-y-4">
        {issues.map((issue) => (
          <li key={issue.id} className="p-4 border rounded shadow">
            <a href={issue.html_url} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline text-lg">
              {issue.title}
            </a>
            <div className="text-sm text-gray-600">
              #{issue.id} opened by{' '}
              <a href={issue.user.html_url} target="_blank" rel="noopener noreferrer" className="text-gray-800 hover:underline">
                {issue.user.login}
              </a>{' '}
              on {new Date(issue.created_at).toLocaleDateString()}
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default IssuesList;
