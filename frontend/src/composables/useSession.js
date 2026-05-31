const STORAGE_KEY = 'pet_chat_thread_id'

export function useSession() {
  function getThreadId() {
    return localStorage.getItem(STORAGE_KEY) || null
  }
  function setThreadId(id) {
    localStorage.setItem(STORAGE_KEY, id)
  }
  function clearThreadId() {
    localStorage.removeItem(STORAGE_KEY)
  }
  return { getThreadId, setThreadId, clearThreadId }
}
