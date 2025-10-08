import React from "react";
import { PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import WordCloud from 'react-d3-cloud';
import './dashboard.css';



const COLORS = ['#5A9CFA', '#FA5A7C', '#FFD166'];

const Dashboard = ({ obj }) => {
  if (!obj) {
    return (
      <div className="dashboard-error" style={{
        color: 'red',
        textAlign: 'center',
        fontWeight: 600,
        marginTop: 40
      }}>
        Error: Data not available or incomplete.<br/>
        Please upload a valid file or check inputs.
        {obj}
      </div>
    );
  }
  // 1. Pie Chart data for sentiment
  const sentimentData = [
    { name: "Positive", value: obj.sentiment?.["+"] || 0 },
    { name: "Negative", value: obj.sentiment?.["-"] || 0 },
    { name: "Neutral",  value: obj.sentiment?.["0"] || 0 }
  ].filter(entry => entry.value > 0);

  // 2. WordCloud Data

  const colors = [
  "#FF6B6B", // red
  "#FFD93D", // yellow
  "#6BCB77", // green
  "#4D96FF", // blue
  "#C77DFF", // purple
  "#FFA41B", // orange
  "#00B8A9", // teal
  "#FF5D8F"  // pink
];

  const randomColor = () =>colors[Math.floor(Math.random()*colors.length)];
  const wordCloudData = obj.wordcount && Object.keys(obj.wordcount).length
    ? Object.entries(obj.wordcount) // yeh par mene array of array banaya h
        .filter(([_, value]) => value > 0) // _ means yeha par text ki jaruat nahi h
        .map(([text, value]) => ({ text, value }))
    : [{ text: "No Data", value: 10 }];

  // 3. Comments List
  const comments = Array.isArray(obj.important_rare) ? obj.important_rare : [];

  return (
    <div className="dashboard-main">
      <h1 className="dashboard-title">Analysis Dashboard</h1>
      <div className="dashboard-row">
        <div className="dashboard-card sentiment-card">
          <h2>Overall Sentiment</h2>
          <ResponsiveContainer width="100%" height={245}>
            <PieChart>
              <Pie
                data={sentimentData}
                dataKey="value"
                nameKey="name"
                cx="50%"
                cy="50%"
                outerRadius={72}
                labelLine={false}
                label={({ name, percent }) =>
                  `${name}: ${(percent * 100).toFixed(0)}%`
                }
              >
                {sentimentData.map((_, idx) => (
                  <Cell key={idx} fill={COLORS[idx % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
              <Legend verticalAlign="bottom" height={24} />
            </PieChart>
          </ResponsiveContainer>
        </div>
        <div className="dashboard-card cloud-card">
          <h2>Frequent Themes</h2>
          <div className="cloud-box">
            <WordCloud
              data={wordCloudData}
              width={window.innerWidth> 600 ? 560 : window.innerWidth*0.95}
              height={window.innerWidth> 600? 260 : 170}
              font="Lato"
              fontWeight="600"
              // fontSize={word => Math.max(20, Math.sqrt(word.value) * 12)}
              // fontSize={(word) => Math.max(12, Math.min(36,Math.sqrt(word.value)*4 ))}
              fontSize={word => Math.max(12, Math.min(36,Math.sqrt(word.value) * 4))} // Use your values directly
              rotate={() => 0} 
              padding={0.8}
              // fill="#5A9CFA"
              fill={randomColor} // colorful words
              //spiral="rectangular"
              spiral="rectangular"
              random={Math.random} // circular pattren like image
            />
          </div>
        </div>
      </div>
      <div className="dashboard-card comments-card">
        <h2>Flagged Unique & Important Comments</h2>
        <div className="comments-list">
          {comments.length ? (
            comments.map((comment, idx) => (
              <div className="comment-item" key={idx}>
                <span className="comment-num">{idx + 1}.</span>
                <span className="comment-text">{comment}</span>
              </div>
            ))
          ) : (
            <div className="comment-empty">No unique comments flagged.</div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;