import apiClient from './client'
import {
  NotebookChatSession,
  NotebookChatSessionWithMessages,
  CreateNotebookChatSessionRequest,
  UpdateNotebookChatSessionRequest,
  SendNotebookChatMessageRequest,
  NotebookChatMessage,
  BuildContextRequest,
  BuildContextResponse,
} from '@/lib/types/api'

export const chatApi = {
  // Session management
  listSessions: async (notebookId: string) => {
    const response = await apiClient.get<NotebookChatSession[]>(
      `/chat/sessions`,
      { params: { notebook_id: notebookId } }
    )
    return response.data
  },

  createSession: async (data: CreateNotebookChatSessionRequest) => {
    const response = await apiClient.post<NotebookChatSession>(
      `/chat/sessions`,
      data
    )
    return response.data
  },

  getSession: async (sessionId: string) => {
    const response = await apiClient.get<NotebookChatSessionWithMessages>(
      `/chat/sessions/${sessionId}`
    )
    return response.data
  },

  updateSession: async (sessionId: string, data: UpdateNotebookChatSessionRequest) => {
    const response = await apiClient.put<NotebookChatSession>(
      `/chat/sessions/${sessionId}`,
      data
    )
    return response.data
  },

  deleteSession: async (sessionId: string) => {
    await apiClient.delete(`/chat/sessions/${sessionId}`)
  },

  // Messaging (synchronous, no streaming) - uses Claude Agent SDK
  sendMessage: async (data: SendNotebookChatMessageRequest) => {
    // Call v2 agent endpoint
    const response = await apiClient.post<{
      session_id: string
      response: string
      messages: Array<{ role: string; content: string }>
      tool_calls_made: number
      cost_usd: number
      provider: string
      error: string | null
    }>(
      `/v2/chat/execute`,
      {
        notebook_id: data.notebook_id,
        session_id: data.session_id,
        message: data.message,
        model_override: data.model_override
      }
    )

    // Map response to expected format
    const messages: NotebookChatMessage[] = response.data.messages.map((msg, idx) => ({
      id: `msg-${Date.now()}-${idx}`,
      type: msg.role === 'user' ? 'human' : 'ai',
      content: msg.content,
      timestamp: new Date().toISOString()
    }))

    return {
      session_id: response.data.session_id,
      messages
    }
  },

  buildContext: async (data: BuildContextRequest) => {
    const response = await apiClient.post<BuildContextResponse>(
      `/chat/context`,
      data
    )
    return response.data
  },
}

export default chatApi
