package pa2;

import org.apache.storm.Config;
import org.apache.storm.StormSubmitter;
import org.apache.storm.topology.TopologyBuilder;
import org.apache.storm.tuple.Fields;

public class ParallelTopology {

	public static void main(String[] args) throws Exception {

		TopologyBuilder builder = new TopologyBuilder();
		builder.setSpout( "twitter_spout", new TwitterSpout() );

		builder.setBolt(" write_bolt", new WriteBolt())
				.setNumTasks(4)
				.fieldsGrouping( "twitter_spout", new Fields("hash"));

		boolean parallel = false;

		parallel = true;
		Config conf = createConfig(parallel);
		StormSubmitter.submitTopology("parallel_twitter",conf,builder.createTopology());
}
	private static Config createConfig(boolean parallel) {
		Config conf = new Config();
		if (parallel)
		{
			conf.setDebug(false);
			conf.setNumWorkers(4);
		}
		else
		{
			conf.setMaxTaskParallelism(1);
		}
		return conf;
	}
}
