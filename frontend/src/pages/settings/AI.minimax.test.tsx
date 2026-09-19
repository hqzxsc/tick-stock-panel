// @vitest-environment jsdom
import { act } from 'react'
import { createRoot } from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { expect, it, vi } from 'vitest'
import { SettingsAIPanel } from './AI'

const state = vi.hoisted(() => ({ data: {} as Record<string, unknown> }))
vi.mock('@/lib/useSharedQueries', () => ({ useSettings: () => state }))
vi.mock('@/lib/useCardFlash', () => ({ useCardFlash: () => false, cardFlashCls: () => '' }))

const cases = [{"label": "MiniMax", "url": "https://api.minimax.io/v1"}, {"label": "MiniMax (China)", "url": "https://api.minimaxi.com/v1"}]
const models = ["MiniMax-M3", "MiniMax-M2.7"]

it.each(cases)('selects $label and preserves a saved alternate model', async ({ label, url }) => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true })
  const client = new QueryClient()
  const host = document.createElement('div')
  const root = createRoot(host)
  const render = () => root.render(<QueryClientProvider client={client}><SettingsAIPanel /></QueryClientProvider>)
  try {
    state.data = { has_ai_key: false, ai_configured: false }
    await act(async () => render())
    const preset = Array.from(host.querySelectorAll('button')).find(button => button.textContent === label)
    expect(preset).toBeDefined()
    await act(async () => preset!.click())
    const values = () => Array.from(host.querySelectorAll('input')).map(input => input.value)
    expect(values()).toContain(url)
    expect(values()).toContain(models[0])
    expect(Array.from(host.querySelectorAll('datalist option')).map(option => option.getAttribute('value'))).toEqual(models)
    state.data = { has_ai_key: true, ai_configured: true, ai_provider: 'openai_compat', ai_base_url: url, ai_openai_model: models[1] }
    await act(async () => render())
    expect(values()).toContain(models[1])
    expect(host.textContent).toContain(label)
  } finally {
    await act(async () => root.unmount())
    client.clear()
  }
})
