import { useState } from 'react'
import axios from 'axios'

function App() {
  const [text, setText] = useState('')
  const [sentencesCount, setSentencesCount] = useState(3)
  const [summary, setSummary] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [modelType, setModelType] = useState('lexrank')

  const handleSummarize = async () => {
    if (!text.trim()) {
      setError('Please enter some text to summarize.')
      return
    }
    
    setError('')
    setLoading(true)
    setSummary('')

    try {
      const response = await axios.post('http://localhost:8000/summarize', {
        text: text,
        sentences_count: sentencesCount,
        model_type: modelType
      }, {
        timeout: 30000 // 30 seconds timeout
      })
      setSummary(response.data.summary)
    } catch (err) {
      console.error(err)
      if (err.code === 'ECONNABORTED') {
        setError('Request timed out. The server took too long to respond.')
      } else {
        setError(err.response?.data?.detail || 'Failed to summarize text. Please try again.')
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="container">
      <header>
        <h1>Text Summarizer</h1>
        <p>Summarize text using LexRank or Google Gemini</p>
      </header>

      <main className="card">
        <div className="input-group">
          <label htmlFor="text-input">Enter your text</label>
          <textarea
            id="text-input"
            placeholder="Paste your text here..."
            value={text}
            onChange={(e) => setText(e.target.value)}
          />
        </div>

        <div className="controls-section">
          
          <div className="model-selector">
            <label>Model:</label>
            <select 
              value={modelType} 
              onChange={(e) => setModelType(e.target.value)}
            >
              <option value="lexrank">LexRank (Local)</option>
              <option value="random_forest">Random Forest (ML)</option>
              <option value="gemini">Gemini (Cloud)</option>
            </select>
          </div>

          <div className="controls">
            <div className="slider-container">
              <label htmlFor="sentences-slider">Sentences: {sentencesCount}</label>
              <input
                id="sentences-slider"
                type="range"
                min="1"
                max="10"
                value={sentencesCount}
                onChange={(e) => setSentencesCount(parseInt(e.target.value))}
              />
            </div>
            
            <button 
              className="btn-primary" 
              onClick={handleSummarize}
              disabled={loading}
            >
              {loading ? <><span className="loading-spinner"></span>Processing...</> : 'Summarize'}
            </button>
          </div>
        </div>

        {error && <div style={{color: '#e53e3e', marginTop: '1rem'}}>{error}</div>}

        {summary && (
          <div className="result-area">
            <label>Summary</label>
            <div className="result-content">
              {summary}
            </div>
          </div>
        )}
      </main>
    </div>
  )
}

export default App
