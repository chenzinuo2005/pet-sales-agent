<script setup>
import { ref, watch, nextTick } from 'vue'
import { useSession } from './composables/useSession.js'
import ChatMessage from './components/ChatMessage.vue'
import ChatInput from './components/ChatInput.vue'
import ImagePreview from './components/ImagePreview.vue'
import Sidebar from './components/Sidebar.vue'
import PawPrints from './components/PawPrints.vue'

const { getThreadId, setThreadId } = useSession()

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
      <div class="header-left">
        <div class="logo-mark" aria-hidden="true">
          <svg width="28" height="28" viewBox="0 0 32 32" fill="none">
            <circle cx="10" cy="12" r="3" fill="currentColor" opacity="0.6"/>
            <circle cx="22" cy="12" r="3" fill="currentColor" opacity="0.6"/>
            <ellipse cx="16" cy="20" rx="6" ry="4.5" fill="currentColor" opacity="0.4"/>
            <path d="M6 22c2 4 5 7 10 7s8-3 10-7" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" opacity="0.5"/>
          </svg>
        </div>
        <h1 class="chat-title">萌宠之家</h1>
      </div>
      <Sidebar @new-session="newSession" @clear-history="clearHistory" />
    </header>

    <main class="chat-main" ref="messageContainer">
      <div class="messages">
        <div v-if="messages.length === 0 && !isStreaming" class="welcome">
          <div class="welcome-hero">
            <div class="welcome-icon">
              <svg width="64" height="64" viewBox="0 0 64 64" fill="none">
                <circle cx="22" cy="24" r="6" fill="var(--color-primary)" opacity="0.3"/>
                <circle cx="42" cy="24" r="6" fill="var(--color-primary)" opacity="0.3"/>
                <ellipse cx="32" cy="40" rx="12" ry="9" fill="var(--color-primary)" opacity="0.15"/>
                <path d="M12 44c4 8 10 14 20 14s16-6 20-14" stroke="var(--color-primary)" stroke-width="2" stroke-linecap="round" opacity="0.35"/>
              </svg>
            </div>
            <h2 class="welcome-heading">欢迎来到萌宠之家</h2>
            <p class="welcome-sub">我是您的专属宠物顾问，上传宠物照片或直接提问吧</p>
          </div>
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
  background: var(--color-canvas);
}

/* ── Header ── */
.chat-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 64px;
  padding: 0 var(--space-lg);
  background: var(--color-canvas);
  border-bottom: 1px solid var(--color-hairline);
  flex-shrink: 0;
  z-index: 10;
}

.header-left {
  display: flex;
  align-items: center;
  gap: var(--space-sm);
}

.logo-mark {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 36px;
  height: 36px;
  border-radius: var(--rounded-sm);
  background: var(--color-surface-soft);
  color: var(--color-primary);
}

.chat-title {
  font-family: var(--font-display);
  font-size: 20px;
  font-weight: 600;
  color: var(--color-ink);
  letter-spacing: -0.18px;
}

/* ── Main scroll area ── */
.chat-main {
  flex: 1;
  overflow-y: auto;
  padding: var(--space-lg) 0;
  background: var(--color-canvas);
}

.messages {
  max-width: 768px;
  margin: 0 auto;
  padding: 0 var(--space-lg);
  display: flex;
  flex-direction: column;
  gap: var(--space-base);
}

/* ── Welcome ── */
.welcome {
  text-align: center;
  padding: var(--space-xxl) var(--space-base);
  animation: fadeInUp 0.5s ease;
}

.welcome-hero {
  margin-bottom: var(--space-xl);
}

.welcome-icon {
  margin-bottom: var(--space-lg);
}

.welcome-heading {
  font-family: var(--font-display);
  font-size: 28px;
  font-weight: 700;
  color: var(--color-ink);
  line-height: 1.43;
  margin-bottom: var(--space-sm);
}

.welcome-sub {
  font-size: 16px;
  font-weight: 400;
  color: var(--color-muted);
  line-height: 1.5;
}

.suggestions {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-sm);
  justify-content: center;
}

.suggestion-chip {
  padding: 10px 20px;
  background: var(--color-canvas);
  border: 1px solid var(--color-hairline);
  border-radius: var(--rounded-full);
  font-size: 14px;
  font-weight: 500;
  color: var(--color-ink);
  line-height: 1.29;
  transition: border-color 0.15s ease, box-shadow 0.15s ease;
}

.suggestion-chip:hover {
  border-color: var(--color-primary);
  box-shadow: var(--shadow-float);
}

/* ── Footer ── */
.chat-footer {
  flex-shrink: 0;
  background: var(--color-canvas);
  border-top: 1px solid var(--color-hairline);
  padding: var(--space-base) var(--space-lg);
  z-index: 10;
}
</style>
