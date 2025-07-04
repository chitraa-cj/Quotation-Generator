<!-- Generated by SpecStory -->

# Fixing async background task in FastAPI (2025-06-09 01:07:26)

_**User**_



---

_**Assistant**_

I notice you haven't provided a specific query, but I can see there's an error trace from your FastAPI application that needs attention. The error occurs when trying to handle a background task in the file download endpoint. Let me help fix this issue.

The error occurs because the `cleanup` function in the `download_file` endpoint is not defined as an async function, but it's being used as a background task. Let me fix this by modifying the code.

I'll edit the file to make the cleanup function async:


---

```diff
      
      # Clean up after download
-     def cleanup():
+     async def cleanup():
          try:
              if os.path.exists(file_path):
```

---

I've modified the `cleanup` function to be async by adding the `async` keyword. This should resolve the TypeError you were seeing where a NoneType couldn't be used in an await expression.

The error occurred because FastAPI's background tasks expect async functions, but we had defined a regular synchronous function. By making the cleanup function async, it can now be properly used as a background task in the FileResponse.

This change should resolve the error you were seeing. The cleanup function will now properly execute after the file download is complete, removing the temporary file and cleaning up the file reference from the `temp_files` dictionary.

Would you like me to help you with anything else regarding your FastAPI application?

---

