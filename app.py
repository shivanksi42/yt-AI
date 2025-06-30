from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, HttpUrl
from typing import List, Dict, Optional
import whisper
import yt_dlp
import os
import tempfile
import uuid
from datetime import datetime
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.tools import Tool
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.question_answering import load_qa_chain
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
import warnings
import logging

# Suppress warnings and configure ChromaDB
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Configure ChromaDB to avoid telemetry issues
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="YouTube AI Assistant with Modern LangChain", version="3.0.0")

load_dotenv()

# Global variables
whisper_model = None
embeddings = None
llm = None

videos_db = {}
vector_stores = {} 
qa_chains = {}

class VideoRequest(BaseModel):
    url: HttpUrl
    
class ChatRequest(BaseModel):
    video_id: str
    question: str
    
class VideoResponse(BaseModel):
    video_id: str
    title: str
    status: str
    message: str
    
class TranscriptResponse(BaseModel):
    video_id: str
    transcript: str
    summary: str
    
class ChatResponse(BaseModel):
    answer: str
    source_documents: List[Dict]
    agent_thought_process: Optional[str] = None

class AgentState(TypedDict):
    question: str
    video_id: str
    context: str
    answer: str
    documents: List[Document]
    intermediate_steps: List[str]
    
def initialize_models():
    """Initialize LangChain components"""
    global whisper_model, embeddings, llm
    
    logger.info("Loading Whisper model...")
    # Use base model and suppress FP16 warning
    whisper_model = whisper.load_model("base", device="cpu")
    
    logger.info("Loading LangChain embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    logger.info("Loading LangChain LLM...")
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        api_key=os.getenv("OPENAI_API_KEY"),
        max_tokens=1000
    )
    
    logger.info("Models loaded successfully!")   

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    try:
        initialize_models()
        logger.info("Application startup completed successfully!")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise

def download_youtube_audio(url: str, output_dir: str) -> tuple[str, str]:
    """Download YouTube video audio"""
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        title = info.get('title', 'Unknown')
        ydl.download([url])
        
        # Find the downloaded audio file
        for file in os.listdir(output_dir):
            if file.endswith('.wav'):
                return os.path.join(output_dir, file), title
                
    raise Exception("Failed to download audio")

def transcribe_audio(audio_path: str) -> Dict:
    """Transcribe audio using Whisper"""
    global whisper_model
    
    if whisper_model is None:
        raise Exception("Whisper model not loaded")
    
    # Suppress FP16 warning by using fp16=False explicitly
    result = whisper_model.transcribe(audio_path, fp16=False)
    
    segments = []
    for segment in result.get('segments', []):
        segments.append({
            'start': segment['start'],
            'end': segment['end'],
            'text': segment['text'].strip()
        })
    
    return {
        'full_text': result['text'],
        'segments': segments,
        'language': result.get('language', 'unknown')
    }

def create_documents_from_transcript(transcript_data: Dict) -> List[Document]:
    """Create LangChain documents from transcript data"""
    documents = []
    
    # Full transcript document
    full_doc = Document(
        page_content=transcript_data['full_text'],
        metadata={
            'type': 'full_transcript',
            'language': transcript_data['language'],
            'content_type': 'overview'
        }
    )
    documents.append(full_doc)
    
    # Individual segment documents
    for segment in transcript_data['segments']:
        if len(segment['text'].strip()) > 10:
            doc = Document(
                page_content=segment['text'],
                metadata={
                    'type': 'segment',
                    'start_time': segment['start'],
                    'end_time': segment['end'],
                    'content_type': 'detailed'
                }
            )
            documents.append(doc)
    
    # Text chunks for better retrieval
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    
    chunks = text_splitter.split_text(transcript_data['full_text'])
    for i, chunk in enumerate(chunks):
        if len(chunk.strip()) > 20:
            doc = Document(
                page_content=chunk,
                metadata={
                    'type': 'chunk',
                    'chunk_id': i,
                    'content_type': 'contextual'
                }
            )
            documents.append(doc)
    
    return documents

def create_vector_store(documents: List[Document], video_id: str) -> Chroma:
    """Create vector store from documents"""
    global embeddings
    
    if embeddings is None:
        raise Exception("Embeddings not loaded")
    
    persist_directory = f"./chroma_db/{video_id}"
    os.makedirs(persist_directory, exist_ok=True)
    
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=f"video_{video_id}"
    )
    
    return vectorstore

def create_summary_chain(documents: List[Document]) -> str:
    """Create summary using modern LangChain summarization with improved prompting"""
    global llm
    
    if llm is None:
        raise Exception("LLM not loaded")
    
    try:
        # Custom prompt for better summarization
        prompt_template = """
        Please provide a comprehensive summary of this video transcript. Focus on:
        1. The main topic or theme
        2. Key points discussed
        3. The overall purpose or message
        4. Any important details or highlights
        
        Transcript content: {text}
        
        Summary:"""
        
        from langchain.prompts import PromptTemplate
        
        # Create custom prompt
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
        
        # Use updated invoke method instead of deprecated run
        summary_chain = load_summarize_chain(
            llm=llm,
            chain_type="stuff",  # Use 'stuff' for better quality with shorter content
            prompt=PROMPT,
            verbose=False
        )
        
        # Filter to only use full transcript for summary
        summary_docs = [doc for doc in documents if doc.metadata.get('type') == 'full_transcript']
        
        if not summary_docs:
            # Fallback to chunks if no full transcript
            summary_docs = [doc for doc in documents if doc.metadata.get('type') == 'chunk'][:3]
        
        if not summary_docs:
            return "No content available for summary"
        
        # Use invoke instead of run
        summary = summary_chain.invoke({"input_documents": summary_docs})
        return summary.get("output_text", "Summary generation failed")
        
    except Exception as e:
        logger.error(f"Error creating summary: {e}")
        # Enhanced fallback summary
        try:
            full_text = " ".join([doc.page_content for doc in documents if doc.metadata.get('type') == 'full_transcript'])
            if not full_text:
                full_text = " ".join([doc.page_content for doc in documents[:5]])
            
            # Simple direct LLM call for summary
            if full_text:
                summary_prompt = f"Summarize this video transcript in 2-3 sentences, focusing on the main topic and key points:\n\n{full_text[:2000]}"
                response = llm.invoke(summary_prompt)
                return response.content
            else:
                return "Unable to generate summary - no transcript content available"
        except Exception as fallback_error:
            logger.error(f"Fallback summary failed: {fallback_error}")
            return "Summary generation failed"

def create_qa_chain(vectorstore: Chroma) -> RetrievalQA:
    """Create QA chain with improved prompt for better context understanding"""
    global llm
    
    if llm is None:
        raise Exception("LLM not loaded")
    
    # Enhanced prompt template
    prompt_template = """
    You are an AI assistant helping users understand video content.You have to act like a teacher and answer every question assuming it is asked by a curious student. Use the following pieces of context to answer the question at the end.
    
    When answering:
    - If the question is about what the video is about, focus on the main theme and topic
    - Include timestamp information when relevant to help users locate specific parts
    - If you don't know the answer based on the context, say you don't know rather than guessing
    - For general questions about the video topic, try to infer the main subject from the available context
    - Do not mention the word "transcript" in your response
    Context: {context}
    
    Question: {question}
    
    Helpful Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Create retrieval QA chain with better retrieval settings
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 8,  # Get more documents for better context
                "fetch_k": 20  # Consider more documents initially
            }
        ),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    
    return qa_chain

def create_search_tools(vectorstore: Chroma, video_id: str) -> List[Tool]:
    """Create modern tools for video search"""
    
    def search_video_content(query: str) -> str:
        """Search through video content"""
        try:
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            docs = retriever.invoke(query)
            
            results = []
            for doc in docs:
                metadata = doc.metadata
                content = doc.page_content
                
                if metadata.get('type') == 'segment':
                    timestamp = f"[{metadata.get('start_time', 0):.1f}s]"
                    results.append(f"{timestamp} {content}")
                else:
                    results.append(content)
            
            return "\n".join(results) if results else "No relevant content found"
        except Exception as e:
            return f"Error searching content: {str(e)}"
    
    def get_video_summary(query: str) -> str:
        """Get video summary"""
        return videos_db.get(video_id, {}).get('summary', 'Summary not available')
    
    def find_specific_timestamp(query: str) -> str:
        """Find specific timestamps for topics"""
        try:
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 10}
            )
            docs = retriever.invoke(query)
            
            timestamp_results = []
            for doc in docs:
                metadata = doc.metadata
                if metadata.get('type') == 'segment':
                    start_time = metadata.get('start_time', 0)
                    end_time = metadata.get('end_time', 0)
                    content = doc.page_content
                    timestamp_results.append(f"[{start_time:.1f}s - {end_time:.1f}s]: {content}")
            
            return "\n".join(timestamp_results[:5]) if timestamp_results else "No timestamps found"
        except Exception as e:
            return f"Error finding timestamps: {str(e)}"
    
    tools = [
        Tool(
            name="search_video_content",
            description="Search through video transcript for specific information",
            func=search_video_content
        ),
        Tool(
            name="get_video_summary",
            description="Get overall summary of the video",
            func=get_video_summary
        ),
        Tool(
            name="find_specific_timestamp",
            description="Find specific timestamps where topics are discussed",
            func=find_specific_timestamp
        )
    ]
    
    return tools

# Modern LangGraph workflow for complex queries
def create_langgraph_workflow(video_id: str):
    """Create LangGraph workflow for complex video analysis"""
    
    def retrieve_context(state: AgentState) -> AgentState:
        """Retrieve relevant context from vector store"""
        question = state["question"]
        video_id = state["video_id"]
        
        if video_id in vector_stores:
            vectorstore = vector_stores[video_id]
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            docs = retriever.invoke(question)
            
            context = "\n".join([doc.page_content for doc in docs])
            state["context"] = context
            state["documents"] = docs
            state["intermediate_steps"].append(f"Retrieved {len(docs)} relevant documents")
        
        return state
    
    def analyze_question_type(state: AgentState) -> AgentState:
        """Analyze question type and route accordingly"""
        question = state["question"].lower()
        
        if any(word in question for word in ['summary', 'about', 'overview', 'main points']):
            state["intermediate_steps"].append("Detected: Summary question")
        elif any(word in question for word in ['when', 'timestamp', 'time', 'moment']):
            state["intermediate_steps"].append("Detected: Timestamp question")
        elif any(word in question for word in ['how', 'step', 'process', 'method']):
            state["intermediate_steps"].append("Detected: Process question")
        else:
            state["intermediate_steps"].append("Detected: General question")
        
        return state
    
    def generate_answer(state: AgentState) -> AgentState:
        """Generate final answer"""
        global llm
        
        question = state["question"]
        context = state.get("context", "")
        
        prompt = f"""
        Based on the video content below, answer the user's question comprehensively.
        
        Video Context:
        {context}
        
        Question: {question}
        
        Answer:"""
        
        try:
            response = llm.invoke(prompt)
            state["answer"] = response.content
            state["intermediate_steps"].append("Generated final answer")
        except Exception as e:
            state["answer"] = f"Error generating answer: {str(e)}"
            state["intermediate_steps"].append(f"Error: {str(e)}")
        
        return state
    
    # Create workflow graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("retrieve", retrieve_context)
    workflow.add_node("analyze", analyze_question_type)
    workflow.add_node("generate", generate_answer)
    
    # Add edges
    workflow.add_edge("retrieve", "analyze")
    workflow.add_edge("analyze", "generate")
    workflow.add_edge("generate", END)
    
    # Set entry point
    workflow.set_entry_point("retrieve")
    
    return workflow.compile()

async def process_video(video_id: str, url: str):
    """Background task to process video with modern LangChain"""
    try:
        global vector_stores, qa_chains
        
        videos_db[video_id]['status'] = 'downloading'
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download and transcribe
            audio_path, title = download_youtube_audio(str(url), temp_dir)
            videos_db[video_id]['title'] = title
            videos_db[video_id]['status'] = 'transcribing'
            
            transcript_data = transcribe_audio(audio_path)
            videos_db[video_id]['status'] = 'processing'
            
            # Create LangChain documents
            documents = create_documents_from_transcript(transcript_data)
            
            # Create vector store
            videos_db[video_id]['status'] = 'indexing'
            vectorstore = create_vector_store(documents, video_id)
            vector_stores[video_id] = vectorstore
            
            # Create summary
            videos_db[video_id]['status'] = 'summarizing'
            summary = create_summary_chain(documents)
            videos_db[video_id]['summary'] = summary
            
            # Create QA chain
            qa_chain = create_qa_chain(vectorstore)
            qa_chains[video_id] = qa_chain
            
            # Store transcript data
            videos_db[video_id]['transcript_data'] = transcript_data
            videos_db[video_id]['status'] = 'completed'
            videos_db[video_id]['processed_at'] = datetime.now().isoformat()
            
    except Exception as e:
        videos_db[video_id]['status'] = 'failed'
        videos_db[video_id]['error'] = str(e)
        logger.error(f"Error processing video {video_id}: {e}")

@app.post("/process-video", response_model=VideoResponse)
async def process_video_endpoint(video_request: VideoRequest, background_tasks: BackgroundTasks):
    """Process a YouTube video"""
    video_id = str(uuid.uuid4())
    
    videos_db[video_id] = {
        'url': str(video_request.url),
        'status': 'queued',
        'created_at': datetime.now().isoformat(),
        'title': ''
    }
    
    background_tasks.add_task(process_video, video_id, video_request.url)
    
    return VideoResponse(
        video_id=video_id,
        title="Processing...",
        status="queued",
        message="Video processing started with modern LangChain pipeline."
    )

@app.get("/video-status/{video_id}")
async def get_video_status(video_id: str):
    """Get processing status"""
    if video_id not in videos_db:
        raise HTTPException(status_code=404, detail="Video not found")
    
    return videos_db[video_id]

@app.get("/transcript/{video_id}", response_model=TranscriptResponse)
async def get_transcript(video_id: str):
    """Get transcript and summary"""
    if video_id not in videos_db or videos_db[video_id]['status'] != 'completed':
        raise HTTPException(status_code=404, detail="Transcript not found or still processing")
    
    video_data = videos_db[video_id]
    transcript_data = video_data['transcript_data']
    
    return TranscriptResponse(
        video_id=video_id,
        transcript=transcript_data['full_text'],
        summary=video_data['summary']
    )

@app.post("/chat", response_model=ChatResponse)
async def chat_with_video(chat_request: ChatRequest):
    """Chat with video using modern QA chain with improved question routing"""
    if chat_request.video_id not in qa_chains:
        raise HTTPException(status_code=404, detail="Video not found or still processing")
    
    try:
        question_lower = chat_request.question.lower()
        video_data = videos_db.get(chat_request.video_id, {})
        
        # Check if this is a summary/overview question
        summary_keywords = ['about', 'summary', 'overview', 'main points', 'topic', 'subject', 'content']
        is_summary_question = any(keyword in question_lower for keyword in summary_keywords)
        
        if is_summary_question and 'summary' in video_data:
            # For summary questions, provide the video summary directly
            return ChatResponse(
                answer=f"This video is about: {video_data['summary']}",
                source_documents=[{
                    "content": video_data['summary'],
                    "type": "summary",
                    "start_time": None,
                    "end_time": None
                }],
                agent_thought_process="Used video summary for overview question"
            )
        
        # For specific questions, use the QA chain
        qa_chain = qa_chains[chat_request.video_id]
        
        # Enhance the query for better retrieval
        enhanced_query = chat_request.question
        if is_summary_question:
            enhanced_query = f"main topic theme overview: {chat_request.question}"
        
        result = qa_chain.invoke({"query": enhanced_query})
        
        # If QA chain doesn't find good results for summary questions, fall back to summary
        if (is_summary_question and 
            (not result["result"] or "don't know" in result["result"].lower()) and 
            'summary' in video_data):
            
            return ChatResponse(
                answer=f"Based on the video content: {video_data['summary']}",
                source_documents=[{
                    "content": video_data['summary'],
                    "type": "summary",
                    "start_time": None,
                    "end_time": None
                }],
                agent_thought_process="Fallback to video summary due to insufficient QA results"
            )
        
        # Format source documents
        source_docs = []
        for doc in result.get("source_documents", []):
            metadata = doc.metadata
            source_info = {
                "content": doc.page_content,
                "type": metadata.get("type", "unknown"),
                "start_time": metadata.get("start_time"),
                "end_time": metadata.get("end_time")
            }
            source_docs.append(source_info)
        
        return ChatResponse(
            answer=result["result"],
            source_documents=source_docs,
            agent_thought_process="Used modern QA chain with retrieval"
        )
        
    except Exception as e:
        logger.error(f"Error in chat processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in QA processing: {str(e)}")



@app.post("/chat-advanced/{video_id}")
async def chat_with_langgraph(video_id: str, question: str):
    """Advanced chat using LangGraph workflow"""
    if video_id not in vector_stores:
        raise HTTPException(status_code=404, detail="Video not found or still processing")
    
    try:
        # Create LangGraph workflow
        workflow = create_langgraph_workflow(video_id)
        
        # Initial state
        initial_state = AgentState(
            question=question,
            video_id=video_id,
            context="",
            answer="",
            documents=[],
            intermediate_steps=[]
        )
        
        # Run workflow
        result = workflow.invoke(initial_state)
        
        return {
            "answer": result["answer"],
            "thought_process": result["intermediate_steps"],
            "documents_used": len(result["documents"])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in LangGraph processing: {str(e)}")

@app.get("/videos")
async def list_videos():
    """List all processed videos"""
    return videos_db

@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy", 
        "models_loaded": whisper_model is not None,
        "langchain_ready": embeddings is not None and llm is not None,
        "openai_api_key_set": bool(os.getenv("OPENAI_API_KEY")),
        "version": "3.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)