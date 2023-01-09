package pa2;

import java.util.Map;
import java.util.concurrent.LinkedBlockingQueue;
import org.apache.storm.spout.SpoutOutputCollector;
import org.apache.storm.task.TopologyContext;
import org.apache.storm.topology.OutputFieldsDeclarer;
import org.apache.storm.topology.base.BaseRichSpout;
import org.apache.storm.tuple.Fields;
import org.apache.storm.tuple.Values;
import org.apache.storm.utils.Utils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import twitter4j.HashtagEntity;
import twitter4j.StallWarning;
import twitter4j.Status;
import twitter4j.StreamListener;
import twitter4j.StatusDeletionNotice;
import twitter4j.StatusListener;
import twitter4j.TwitterStream;
import twitter4j.TwitterStreamFactory;
import twitter4j.auth.AccessToken;
import twitter4j.conf.ConfigurationBuilder;
import twitter4j.FilterQuery;

public class TwitterSpout extends BaseRichSpout {

	TwitterStream twitterStream;

	String consumerKey = "KpHRi90CRut4JE3ilESpfZsxd";
	String consumerSecret = "00pCIw58twUpHUwIk1UMuS0stY5NiIvnt1V3xrJAE6FBBKCpF7" ;
	String accessToken = "1379146026523582464-4loRoAVBJ9WD7HTPFScmbrl7IOxvUh";
	String accessTokenSecret = "tN9LeVbQ2YEF1c4q8Or8vbYcacNSoTiCVuFWM8MySecGH" ;

	SpoutOutputCollector collector;
	LinkedBlockingQueue<Status> queue = null;

	@Override
	public void open(Map<String, Object> conf, TopologyContext context,
			SpoutOutputCollector collector)
	{
		queue = new LinkedBlockingQueue<Status>(1000);
		this.collector = collector;

		StatusListener statusListener = new StatusListener()
		{

			@Override
			public void onStatus(Status status) {
				queue.offer(status);
			}

			@Override
			public void onException(Exception ex) {}

			@Override
			public void onDeletionNotice(
					StatusDeletionNotice statusDeletionNotice) {}

			@Override
			public void onTrackLimitationNotice(int numberOfLimitedStatuses) {}

			@Override
			public void onScrubGeo(long userId, long upToStatusId) {}

			@Override
			public void onStallWarning(StallWarning warning) {}

		};

		this.twitterStream = new TwitterStreamFactory(
				new ConfigurationBuilder().setJSONStoreEnabled(true).build())
				.getInstance();
		this.twitterStream.addListener(statusListener);
		this.twitterStream.setOAuthConsumer(consumerKey, consumerSecret);
		AccessToken token =
				new AccessToken(accessToken, accessTokenSecret);
		this.twitterStream.setOAuthAccessToken(token);

		FilterQuery query = new FilterQuery();
		query.track(new String[]{"Machine Learning","原神"});

		this.twitterStream.filter(new FilterQuery().language("en"));
		this.twitterStream.sample();
	}

	@Override
	public void nextTuple() {
		Status status = queue.poll();
		if (status == null)
		{
			Utils.sleep(100);
		} else
		{
			for (HashtagEntity entity : status.getHashtagEntities())
			{
				if (status.getText().matches("\\A\\p{ASCII}*\\z"))
				{
					collector.emit(new Values(status.getText(),entity.getText()));//emit tuple to Bolt
				}
			}

		
		}
	}

	@Override
	public void close()
	{
		twitterStream.shutdown();
	}

	@Override
	public void declareOutputFields(OutputFieldsDeclarer declarer)	 {
		declarer.declare(new Fields("hash","tag"));
	}

}
