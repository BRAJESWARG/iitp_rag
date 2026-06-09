/// <reference types="vite/client" />

// CSS module declarations
declare module '*.css' {
  const content: Record<string, string>;
  export default content;
}

// SVG declarations
declare module '*.svg' {
  const content: string;
  export default content;
}

// Image declarations
declare module '*.png' {
  const content: string;
  export default content;
}
