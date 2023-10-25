Reveal.initialize({
    autoPlayMedia: true,
    slideNumber: 'h/v',
    center: true,
    hash: true,
    mathjax2: {
        config: 'TeX-AMS_HTML-full',
        TeX: {
            Macros: {
                R: '\\mathbb{R}',
                set: ['\\left\\{#1 \\; ; \\; #2\\right\\}', 2]
            }
        }
    },
    plugins: [RevealHighlight, RevealMarkdown, RevealMath.MathJax2, RevealMenu],
    slideNumber: true
});