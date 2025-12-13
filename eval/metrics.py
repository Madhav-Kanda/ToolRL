def parse_pytest_summary(output: str):
    lines = [l for l in output.splitlines() if " passed" in l or " failed" in l or " error" in l]
    return "\n".join(lines[-3:])


