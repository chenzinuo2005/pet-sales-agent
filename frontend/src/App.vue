<script setup>
import { ref, watch, nextTick } from 'vue'
import { useSession } from './composables/useSession.js'
import ChatMessage from './components/ChatMessage.vue'
import ChatInput from './components/ChatInput.vue'
import ImagePreview from './components/ImagePreview.vue'
import Sidebar from './components/Sidebar.vue'
import PawPrints from './components/PawPrints.vue'

const { getThreadId, setThreadId, clearThreadId } = useSession()

const messages = ref([])
const threadId = ref(getThreadId())
const isStreaming = ref(false)
const streamingContent = ref('')
const selectedImage = ref(null)
const selectedImageUrl = ref(null)
const messageContainer = ref(null)

async function ensureSession() {
  if (!threadId.value) {
    const res = await fetch('/api/session', {
      method: 'POST',
      signal: AbortSignal.timeout(10000),
    })
    if (!res.ok) throw new Error('创建会话失败')
    const data = await res.json()
    threadId.value = data.thread_id
    setThreadId(data.thread_id)
  }
}

function scrollToBottom() {
  nextTick(() => {
    if (messageContainer.value) {
      messageContainer.value.scrollTop = messageContainer.value.scrollHeight
    }
  })
}

watch(() => messages.value.length, scrollToBottom)
watch(streamingContent, scrollToBottom)

async function sendMessage(_text) {
  const text = _text.trim() || (selectedImage.value ? '这是什么品种？' : '')
  if (!text || isStreaming.value) return

  try {
    await ensureSession()

    messages.value.push({ role: 'user', content: text })
    isStreaming.value = true
    streamingContent.value = ''

    let res
    if (selectedImage.value) {
      const formData = new FormData()
      formData.append('image', selectedImage.value)
      formData.append('message', text)
      formData.append('thread_id', threadId.value)
      res = await fetch('/api/chat/upload', {
        method: 'POST',
        body: formData,
        signal: AbortSignal.timeout(60000),
      })
    } else {
      res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: AbortSignal.timeout(60000),
        body: JSON.stringify({
          message: text,
          thread_id: threadId.value,
        }),
      })
    }

    if (!res.ok) throw new Error('请求失败')

    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    let currentEvent = ''
    let currentData = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''

      for (const line of lines) {
        if (!line.trim()) {
          // Empty line = end of SSE event
          if (currentEvent === 'token' && currentData) {
            streamingContent.value += currentData
          } else if (currentEvent === 'done') {
            messages.value.push({ role: 'assistant', content: streamingContent.value })
            streamingContent.value = ''
          } else if (currentEvent === 'error') {
            console.error('SSE error:', currentData)
            messages.value.push({
              role: 'assistant',
              content: '抱歉，出了点小问题，请稍后再试...',
            })
          }
          currentEvent = ''
          currentData = ''
          continue
        }

        if (line.startsWith('event: ')) {
          currentEvent = line.slice(7).trim()
        } else if (line.startsWith('data: ')) {
          currentData = line.slice(6).trim()
        }
      }
    }
  } catch (err) {
    console.error(err)
    messages.value.push({
      role: 'assistant',
      content: '抱歉，出了点小问题，请稍后再试...',
    })
  } finally {
    isStreaming.value = false
    selectedImage.value = null
    selectedImageUrl.value = null
  }
}

function onImageSelected(file) {
  selectedImage.value = file
  if (selectedImageUrl.value) {
    URL.revokeObjectURL(selectedImageUrl.value)
  }
  selectedImageUrl.value = URL.createObjectURL(file)
}

async function newSession() {
  try {
    const res = await fetch('/api/session', { method: 'POST' })
    if (!res.ok) throw new Error('创建会话失败')
    const data = await res.json()
    threadId.value = data.thread_id
    setThreadId(data.thread_id)
    messages.value = []
  } catch (err) {
    console.error(err)
  }
}

async function clearHistory() {
  if (!threadId.value) return
  try {
    const res = await fetch(`/api/history/${threadId.value}`, {
      method: 'DELETE',
    })
    if (!res.ok) throw new Error('清除历史失败')
    messages.value = []
  } catch (err) {
    console.error(err)
  }
}
</script>

<template>
  <div class="chat-app">
    <PawPrints />
    <header class="chat-header">
      <h1 class="chat-title">萌宠之家</h1>
      <Sidebar @new-session="newSession" @clear-history="clearHistory" />
    </header>

    <main class="chat-main" ref="messageContainer">
      <div class="messages">
        <div v-if="messages.length === 0 && !isStreaming" class="welcome">
          <div class="welcome-icon">🐾</div>
          <h2>欢迎来到萌宠之家</h2>
          <p>我是您的专属宠物顾问，有什么可以帮您的？</p>
          <div class="suggestions">
            <button
              v-for="q in ['推荐适合新手的宠物', '狗狗掉毛怎么办', '猫咪需要打哪些疫苗']"
              :key="q"
              class="suggestion-chip"
              @click="sendMessage(q)"
            >{{ q }}</button>
          </div>
        </div>

        <ChatMessage
          v-for="(msg, i) in messages"
          :key="i"
          :role="msg.role"
          :content="msg.content"
          :isLast="i === messages.length - 1"
          :isStreaming="false"
        />
        <ChatMessage
          v-if="isStreaming && streamingContent"
          role="assistant"
          :content="streamingContent"
          :isLast="true"
          :isStreaming="true"
        />
        <ChatMessage
          v-if="isStreaming && !streamingContent"
          role="assistant"
          content=""
          :isLast="true"
          :isStreaming="true"
        />
      </div>
    </main>

    <footer class="chat-footer">
      <ImagePreview
        v-if="selectedImageUrl"
        :url="selectedImageUrl"
        @remove="selectedImage = null; selectedImageUrl = null"
      />
      <ChatInput
        :disabled="isStreaming"
        :hasImage="!!selectedImage"
        @send="sendMessage"
        @image-selected="onImageSelected"
      />
    </footer>
  </div>
</template>

<style scoped>
.chat-app {
  position: relative;
  display: flex;
  flex-direction: column;
  height: 100vh;
  overflow: hidden;
}

.chat-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 20px;
  background: var(--color-surface);
  border-bottom: 1px solid var(--color-border);
  flex-shrink: 0;
  z-index: 10;
}

.chat-title {
  font-family: var(--font-display);
  font-size: 22px;
  color: var(--color-primary);
  font-weight: 400;
}

.chat-main {
  flex: 1;
  overflow-y: auto;
  padding: 16px 0;
}

.messages {
  max-width: 800px;
  margin: 0 auto;
  padding: 0 16px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.welcome {
  text-align: center;
  padding: 48px 16px;
  animation: fadeInUp 0.6s ease;
}

.welcome-icon {
  font-size: 56px;
  margin-bottom: 12px;
}

.welcome h2 {
  font-family: var(--font-display);
  font-size: 24px;
  color: var(--color-primary);
  margin-bottom: 8px;
}

.welcome p {
  color: var(--color-text-muted);
  font-size: 15px;
  margin-bottom: 24px;
}

.suggestions {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  justify-content: center;
}

.suggestion-chip {
  padding: 8px 18px;
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-btn);
  font-size: 14px;
  color: var(--color-text);
  transition: all 0.2s ease;
}

.suggestion-chip:hover {
  border-color: var(--color-accent);
  color: var(--color-accent);
  background: #FFF5EC;
}

.chat-footer {
  flex-shrink: 0;
  border-top: 1px solid var(--color-border);
  background: var(--color-surface);
  padding: 12px 20px;
  z-index: 10;
}
</style>
