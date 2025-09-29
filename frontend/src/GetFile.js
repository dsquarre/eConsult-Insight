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
            const endpoint = "http://0.0.0.0:8000/upload/";
            const response = await fetch(endpoint, {
                method: "POST",
                body: formData
            });

            if (response.ok) {
                console.log("File submitted successfully. Awaiting processing.");
            } else {
                const errorText = await response.text();
                throw new Error(`Server failed to process file: ${errorText}`);
            }
        } catch (err) {
            console.error("Submission error:", err);
            setError(`Failed to submit file. Please check your network. (${err.message})`);
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