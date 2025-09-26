import { useState, useEffect } from "react";
import './App.css'
import Result from "./result";


const GetLink = () => {
    const [link, setLink] = useState('');
    const [isProcessing, setIsProcessing] = useState(false);
    const [processedData, setProcessedData] = useState(null);
    const [error, setError] = useState(null);
    
    const handleSubmit = async (event) => {
        event.preventDefault();
        
        if (!link) {
            setError("Please enter a valid link.");
            return;
        }

        setError(null);
        setIsProcessing(true); 
        setProcessedData(null); 
        
        console.log("Submitting link for analysis:", link);
        
        const formData = new FormData();
        formData.append("text", link);

        try {
            const endpoint = "http://0.0.0.0:8000/upload/";
            const response = await fetch(endpoint, {
                method: "POST",
                body: formData
            });

            if (response.ok) {
                console.log("Link submitted successfully. Awaiting processing.");
            } else {
                const errorText = await response.text();
                throw new Error(`Server failed to process link: ${errorText}`);
            }
        } catch (err) {
            console.error("Submission error:", err);
            setError(`Failed to submit link. Please check your network. (${err.message})`);
            setIsProcessing(false); // Stop processing state on failure
        }
    };

    useEffect(() => {
        if (isProcessing) {
            const fetchProcessedText = async () => {
                try {
                    const response = await fetch('http://0.0.0.0:8000/result/');
                    
                    if (!response.ok) {
                         throw new Error(`Failed to fetch processed result: ${response.statusText}`);
                    }
                    
                    const data = await response.json(); 
                    
                    // 'data' is an object like: 
                    // {indexed clean comments:[] }
                    setProcessedData(data); 
                    setIsProcessing(false); 
                    console.log("Structured data fetched:", data);

                } catch (err) {
                    console.error('Error fetching structured data:', err);
                    setError(`Error retrieving result. Expected JSON data but received an error: ${err.message}`);
                    setIsProcessing(false);
                }
            };
            
            fetchProcessedText();
        }
    }, [isProcessing]);


    if (error) {
        return (
            <div className="error-message">
                <h1>Error</h1>
                <p>{error}</p>
                <button onClick={() => { setIsProcessing(false); setError(null); setLink(''); }}>Try Again</button>
            </div>
        );
    }
    
    if (isProcessing) {
        return (
            <div className="processing">
                <h1>Please wait while we process your comments...</h1>
                <div className="spinner" /> 
            </div>
        );
    }

    if (processedData) {
        return (
            <div>
                <Result data={processedData} />
            </div>
        );
    }

    return (
        <div>
           <div> <h1> Paste the link of website that has the comments </h1></div>
            
            <form onSubmit={handleSubmit}>
                <div className = "textbox">
                <input
                    type="url"
                    value={link}
                    onChange={(e) => setLink(e.target.value)}
                    placeholder="e.g., https://example.com/page-with-comments"
                    required
                />
                </div>
                <button type="submit">Analyze Comments</button>
            </form>
            
        </div>
    );
};

export default GetLink;