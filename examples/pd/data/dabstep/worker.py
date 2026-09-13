"""Persistent Python session. Must run ONLY inside the restricted container."""
import contextlib
import io
import json
import sys
import time
import traceback


class LimitedOutput(io.StringIO):
    def write(self, value):
        remaining = max(0, 32000 - self.tell())
        super().write(value[:remaining])
        return len(value)


class FinalAnswer(BaseException):
    pass


def main():
    scope = {"__name__": "__main__"}
    result = {}
    def final_answer(answer):
        result["answer"] = str(answer)
        raise FinalAnswer()
    scope["final_answer"] = final_answer
    protocol = sys.stdout
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        output = LimitedOutput()
        result.clear()
        start = time.perf_counter()
        error = None
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            try:
                exec(compile(request["code"], "<model-tool>", "exec"), scope)
            except FinalAnswer:
                pass
            except BaseException:
                error = traceback.format_exc()[-4000:]
        print(json.dumps({"output": result.get("answer"), "is_final_answer": "answer" in result,
                          "logs": output.getvalue(), "error": error,
                          "execution_seconds": time.perf_counter() - start}), file=protocol, flush=True)


if __name__ == "__main__":
    main()
