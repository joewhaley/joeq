package Main;

public class SimpleCrash implements jq.CrashInterface {
    public void die() {
	Run_Time.DebugInterface.die(-1);
    }
}
