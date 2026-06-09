// ─────────────────────────────────────────────────────────────────
// App.tsx — Root application component
// ─────────────────────────────────────────────────────────────────
import './App.css';
import { useRAG } from './hooks/useRAG';
import { Header } from './components/Header';
import { Sidebar } from './components/Sidebar';
import { ChatWindow } from './components/ChatWindow';

function App() {
  const {
    messages,
    pipeline,
    status,
    isLoading,
    isIngesting,
    sendQuery,
    ingestFiles,
    clearMessages,
  } = useRAG();

  return (
    <div className="app-layout">
      <Header
        status={status}
        onClear={clearMessages}
      />
      <Sidebar
        status={status}
        pipeline={pipeline}
        onIngest={ingestFiles}
        isIngesting={isIngesting}
      />
      <ChatWindow
        messages={messages}
        pipeline={pipeline}
        isLoading={isLoading}
        isInitialized={status?.initialized ?? false}
        onSend={sendQuery}
      />
    </div>
  );
}

export default App;
