package Run_Time;

public class DebugImpl implements Debug.Delegate {

    public void write(byte[] msg, int size) {
        SystemInterface.debugwrite(msg, size);
    }

    public void write(String msg) {
        SystemInterface.debugwrite(msg);
    }

    public void writeln(byte[] msg, int size) {
        SystemInterface.debugwriteln(msg, size);
    }

    public void writeln(String msg) {
        SystemInterface.debugwriteln(msg);
    }

    public void die(int code) {
        if (code != 0)
            Debugger.OnlineDebugger.debuggerEntryPoint();
        SystemInterface.die(code);
    }

}
