package Scheduler;

public abstract class ThreadUtils {
    public static jq_Thread getJQThread(java.lang.Thread t) {
	return _delegate.getJQThread(t);
    }
    static interface Delegate {
	jq_Thread getJQThread(java.lang.Thread t);
    }

    private static Delegate _delegate;
    static {
	/* Set up delegates. */
	_delegate = null;
	boolean nullVM = Main.jq.nullVM || System.getProperty("joeq.nullvm") != null;
	if (!nullVM) {
	    _delegate = attemptDelegate("Scheduler.FullThreadUtils");
	}
	if (_delegate == null) {
	    _delegate = new Scheduler.HostedThreadUtils();
	}
    }

    private static Delegate attemptDelegate(String s) {
	String type = "thread util delegate";
        try {
            Class c = Class.forName(s);
            return (Delegate)c.newInstance();
        } catch (java.lang.ClassNotFoundException x) {
            System.err.println("Cannot find "+type+" "+s+": "+x);
        } catch (java.lang.InstantiationException x) {
            System.err.println("Cannot instantiate "+type+" "+s+": "+x);
        } catch (java.lang.IllegalAccessException x) {
            System.err.println("Cannot access "+type+" "+s+": "+x);
        }
	return null;
    }

}
