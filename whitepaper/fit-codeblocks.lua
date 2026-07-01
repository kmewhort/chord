-- Pandoc Lua filter: scale over-wide code blocks to the text width for the PDF.
--
-- The whitepaper's Appendix D.1 architecture diagram is ASCII art ~124 columns
-- wide — wider than any letter/A4 text block at a readable monospace size. Rather
-- than guess a font size, capture wide blocks into a save-box and \resizebox them
-- to exactly the text width (and only wide ones; normal code is left untouched).
-- A verbatim environment cannot appear inside a macro argument, so we fill an
-- `lrbox` first (the \diagbox declared in header.tex), then scale the box.
--
-- Only affects the LaTeX/PDF path; HTML/GitHub rendering is unchanged.

local WIDTH_THRESHOLD = 90  -- columns; blocks wider than this get scaled

local function max_line_width(text)
  local mx = 0
  for line in (text .. "\n"):gmatch("(.-)\n") do
    -- count Unicode codepoints, not bytes, so box-drawing chars count as 1
    local n = select(2, line:gsub("[^\128-\191]", ""))
    if n > mx then mx = n end
  end
  return mx
end

function CodeBlock(el)
  if FORMAT ~= "latex" and FORMAT ~= "beamer" then
    return nil
  end
  if max_line_width(el.text) <= WIDTH_THRESHOLD then
    return nil
  end
  local latex =
    "\\begin{center}%\n" ..
    "\\begin{lrbox}{\\diagbox}%\n" ..
    "\\begin{BVerbatim}\n" ..
    el.text .. "\n" ..
    "\\end{BVerbatim}\n" ..
    "\\end{lrbox}%\n" ..
    "\\resizebox{\\textwidth}{!}{\\usebox{\\diagbox}}%\n" ..
    "\\end{center}"
  return pandoc.RawBlock("latex", latex)
end
