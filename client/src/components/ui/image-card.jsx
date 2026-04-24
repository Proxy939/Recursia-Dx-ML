import React, { useState } from "react"
import { cn } from "@/lib/utils"

/**
 * Rewrite .dcm URLs to their .png equivalent.
 * The backend DICOM middleware serves the PNG, but browsers cache the
 * old failed request. Rewriting the extension avoids that entirely.
 */
function normalizeSrc(src) {
  if (!src) return src
  // If URL ends with .dcm, request the pre-converted .png instead
  if (src.match(/\.dcm(\?.*)?$/i)) {
    return src.replace(/\.dcm(\?.*)?$/i, '.png')
  }
  return src
}

export function ImageCard({
  src,
  alt,
  className,
  ...props
}) {
  const normalized = normalizeSrc(src)
  const [currentSrc, setCurrentSrc] = useState(normalized)
  const [retried, setRetried] = useState(false)

  const handleLoad = () => {
    console.log('✅ Image loaded:', currentSrc)
  }

  const handleError = () => {
    console.error('❌ Image failed to load:', currentSrc)
    // One retry with cache-busting in case PNG wasn't ready yet
    if (!retried && normalized) {
      setRetried(true)
      setCurrentSrc(`${normalized}?t=${Date.now()}`)
    }
  }

  return (
    <img
      src={currentSrc}
      alt={alt}
      className={cn("rounded-lg border", className)}
      onLoad={handleLoad}
      onError={handleError}
      {...props}
    />
  )
}
