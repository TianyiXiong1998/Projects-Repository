package pa2;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.time.Instant;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.stream.Collectors;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.storm.Config;
import org.apache.storm.Constants;
import org.apache.storm.task.OutputCollector;
import org.apache.storm.task.TopologyContext;
import org.apache.storm.topology.OutputFieldsDeclarer;
import org.apache.storm.topology.base.BaseRichBolt;
import org.apache.storm.tuple.Tuple;

public class WriteBolt extends BaseRichBolt {
	private final Map<String, String> item_content = new HashMap<String, String>();
	private BufferedWriter writer;
	private int emitFrequency = 10;


	@Override
	public Map<String, Object> getComponentConfiguration()
	{
		Config conf = new Config();
		conf.put(Config.TOPOLOGY_TICK_TUPLE_FREQ_SECS, emitFrequency);
		return conf;
	}
	private void insert(Tuple input)
	{
		String s = input.getStringByField("hash");//get key set in the hash field
		String words[] = s.split(" ");
		String value = words[1];
		item_content.put(s,value);
	}
	@Override
	public void execute(Tuple tuple)
	{

			// do write
			StringBuilder write_file = new StringBuilder(Instant.now().toString());
			write_file.append("<");
			String s = tuple.getStringByField("hash");
			String tag = tuple.getStringByField("tag");
			write_file.append("text:").append(s).append("-").append("tag:").append(tag);
			write_file.append(">").append("\n");
/*			Map<String, String> output = item_content;
			item_content.clear();
			for (Entry<String,String>data_entry:output.entrySet())
			{
				write_file.append(data_entry.getKey()).append("-").append(data_entry.getValue()).append( "><" );
			}*/

			try
			{
				writer.write(write_file + "\n");
				writer.flush();
			} catch (IOException e)
			{
				e.printStackTrace();
			}


	}


	@Override
	public void prepare(Map<String, Object> topoConf, TopologyContext context,
			OutputCollector outputCollector) {
		try
		{
			writer = new BufferedWriter(new FileWriter( "/s/chopin/k/grad/xiongty/PA2/output.txt", true ));
		} catch (IOException e)
		{
			e.printStackTrace();
		}
	}

	@Override
	public void declareOutputFields(OutputFieldsDeclarer declarer) {}

}
