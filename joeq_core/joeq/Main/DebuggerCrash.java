package Main;

public class DebuggerCrash implements jq.CrashInterface {
    public void die() {
	if (jq.RunningNative) {
            Debug.OnlineDebugger.debuggerEntryPoint();
            //new InternalError().printStackTrace();
        }
        Run_Time.DebugInterface.die(-1);
    }
}
