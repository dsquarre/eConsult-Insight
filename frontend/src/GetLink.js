import { useState, useEffect } from "react";
import './App.css'
import DashBoard from "./dashboard";


const GetLink = () => {
    const [link, setLink] = useState('');
    const [isProcessing, setIsProcessing] = useState(false);
    const [result, setresult] = useState(null);
    const [error, setError] = useState(null);
    
    const handleSubmit = async (event) => {
        event.preventDefault();
        
        if (!link) {
            setError("Please enter a valid link.");
            return;
        }

        setError(null);
        setIsProcessing(true); 
        setresult(null); 
        
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
            const fetchProcessedResult = async () => {
                try {
                    const response = await fetch('http://0.0.0.0:8000/result/');
                    
                    if (!response.ok) {
                         throw new Error(`Failed to fetch processed result: ${response.statusText}`);
                    }
                    
                    const result = await response.json(); 
                    
                    // 'result' is a JSON file like: 
                    // {wordcount:{word:count}, sentiment:{+:val, -:val, 0:val}, important_rare:[comment1,...]}
                    setresult(result); 
                    setIsProcessing(false); 
                    console.log("Structured data fetched:", result);

                } catch (err) {
                    console.error('Error fetching structured data:', err);
                    setError(`Error retrieving result. Expected JSON data but received an error: ${err.message}`);
                    setIsProcessing(false);
                }
            };
            
            fetchProcessedResult();
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

    if (result) {
        return (
            <div>
                <DashBoard obj={result} />
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