package Scheduler;

import Clazz.jq_Class;
import Run_Time.Reflection;

public class HostedThreadUtils implements ThreadUtils.Delegate {
    public jq_Thread getJQThread(java.lang.Thread t) {
	jq_Class k = Bootstrap.PrimordialClassLoader.getJavaLangThread();
	Clazz.jq_InstanceField f = k.getOrCreateInstanceField("jq_thread", "LScheduler/jq_Thread;");
	return (jq_Thread)Reflection.getfield_A(t, f);
    }    
}
