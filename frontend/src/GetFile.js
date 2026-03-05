import { useRef,useState, useEffect } from "react";
import './App.css'
import DashBoard from "./dashboard";


const GetFile = () => {
    const [file, setFile] = useState('');
    const [isProcessing, setIsProcessing] = useState(false);
    const [result, setresult] = useState(null);
    const [error, setError] = useState(null);
    const inputRef = useRef();
    const handleDragOver = (event) =>{
        event.preventDefault();
    };
    const handleDrop = (event) => {
        event.preventDefault();
        setFile(event.dataTransfer.files[0])
    };
    const handleClick = async (event) => {
        event.preventDefault();
        
        if (!file) {
            setError("Please enter a valid file.");
            return;
        }

        setError(null);
        setIsProcessing(true); 
        setresult(null); 
        
        console.log("Submitting file for analysis:", file);
        
        const formData = new FormData();
        formData.append("file", file);

        try {
            const endpoint = "http://localhost:8000/upload/";
            const response = await fetch(endpoint, {
                method: "POST",
                body: formData
            });

            if (response.ok) {
                console.log("File submitted successfully. Fetching results...");
                
                const resultResponse = await fetch('http://localhost:8000/result/');
                
                if (!resultResponse.ok) {
                     throw new Error(`Failed to fetch processed result: ${resultResponse.statusText}`);
                }
                
                const result = await resultResponse.json(); 
                setresult(result); 
                console.log("Structured data fetched:", result);
            } else {
                const errorText = await response.text();
                let errorMessage = errorText;
                try {
                    const errorJson = JSON.parse(errorText);
                    if (errorJson.error) errorMessage = errorJson.error;
                } catch (e) {}
                throw new Error(errorMessage);
            }
        } catch (err) {
            console.error("Submission error:", err);
            setError(`Failed to submit file. (${err.message})`);
        } finally {
            setIsProcessing(false);
        }
    };

    if (error) {
        return (
            <div className="error-message">
                <h1>Error</h1>
                <p>{error}</p>
                <button onClick={() => { setIsProcessing(false); setError(null); setFile(''); }}>Try Again</button>
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

    if (file){
        return (
        <div className="uploads">
            <h1>
                File: {file.name}
            </h1>
            <div className="actions">
                <button onClick={() => setFile(null)}>Cancel</button>
                <button onClick={handleClick }>Upload</button>
            </div>
        </div>
    )
    }
    return (
        <div 
                className="dropzone"
                onDragOver={handleDragOver}
                onDrop={handleDrop}
            >
                <h1> Drag and Drop Files to Upload</h1>
                <h1>OR</h1>
                <input 
                    type="file"
                    onChange = {(event) => setFile(event.target.files[0])}
                    hidden
                    ref={inputRef}
                />
            
                <button onClick={() => inputRef.current.click()}>Select Files</button>
            </div>
    );
};

export default GetFile;